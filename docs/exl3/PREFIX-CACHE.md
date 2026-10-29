# Prefix cache agentique — le quadriptyque v2

**Date :** 2026-10-26 · **Lane :** `vllm-exl3` · **Statut :** opt-in mesuré,
sauf le no-store et la métrique (observabilité sans effet de bord).

Le prefix cache de cette lane est bon en mono-session (85 % de hit, 4,1× sur
les follow-ups). Le risque réel est ailleurs : en **multi-session / co-batch**,
un cache KDA hybride peut retomber à **0 %** sans erreur, parce qu'un groupe
KV manque un checkpoint et veto tous les autres. Les overlays de ce bundle
ferment les quatre trous connus.

## Pourquoi 0 % en multi-session

GLM-5.3-Flash est hybride : les couches KDA cachent un **état récurrent**, pas
des tokens. vLLM dimensionne le cache en pages de **3584 tokens** et
`mamba_cache_mode=align` ne pose un checkpoint KDA qu'à une frontière de page.
Le coordinateur hybride exige que **chaque** groupe KV hit ; un seul groupe qui
manque annule le hit des autres. Les compteurs lisent alors zéro, sans log.

Trois consequences :

1. `MAX_NUM_BATCHED_TOKENS` doit être ≥ 3584 (la lane est à 7168, donc OK) ;
2. les hits sont alignés sur 3584 : un prompt < 3,6k ne peut jamais hit ;
3. le drafter DFlash2 (SlidingWindow, 64 blocs) hashe 33 ids par frontière et
   les remet dans le LRU **avant** les blocs MLA/mamba qui portent le hit :
   priorité inversée, 87 % du pool part dans un groupe dont le hit est jeté par
   le `min()` hybride.

`patch_hybrid_prefix_hit.py` (déjà dans la lane) corrige le `min()` et le
`eagle_group_ids` ; il ne corrige pas la rétention fine ni le veto
`KpoolTailManager`.

## Les quatre overlays

| Fichier | Effet | Risque | Défaut |
|---|---|---|---|
| `patch_apc_per_group_retention.py` | rétention **par groupe** : MLA/mamba dense, drafter SWA aux frontières atteignables | faible ; change le LRU, pas le calcul | **on** (`GLM53_APC_PER_GROUP_RETENTION=1`) |
| `patch_fine_grained_apc.py` | exempte les managers non-participants (`KpoolTailManager`) du veto de hash fin → hits à **64 tokens** | change la granularité ; la référence mesure **−2,7 % structured** | **off** (`GLM53_FINE_GRAINED_APC=0`) |
| `patch_apc_no_store.py` | opt-out d'écriture par requête (`skip_writing_prefix_cache`) : un batch/eval ne pollue plus le LRU | faible ; read-hit conservé | **on** (`GLM53_APC_NO_STORE=1`) |
| `patch_prefix_cache_sparse_miss_metric.py` | exporte `vllm:prefix_cache_sparse_retention_misses` | aucun (métrique seule) | **on** (`GLM53_PREFIX_CACHE_SPARSE_MISS_METRIC=1`) |

Ordre d'installation : `patch_hybrid_prefix_hit.py` →
`patch_apc_per_group_retention.py` → `patch_fine_grained_apc.py` →
`patch_prefix_cache_sparse_miss_metric.py` → `patch_apc_no_store.py`.
Le launcher applique cet ordre sur les deux rangs.

## Ce que ça change concrètement

Mesures de la référence Apache-2.0 (à reproduire sur ce kit, pas des claims) :

| Forme | Avant | Après |
|---|---|---|
| 2×68k, round 2 | **0 %** (163 s) | **100 %** (1,2 s) |
| 4×60k concurrent ×3 | **0 %** (288 s) | **98,7 %** (16,5 s) |
| Prompt 6,5k, follow-up | 3 584 tokens réutilisés | **6 464** (96,2 %) |
| Prompt 2,6k | 0 | **2 816** |
| Replay 110k | 98 % | 98 % (inchangé) |
| Shared prefix, longueur différente | 0 % | **37–49 %** |

Le gain n'est pas un tok/s de décode : c'est le **TTFT des tours suivants**.
Un agent qui relit 60k tokens passe de dizaines de secondes à ~1 s.

## Mesurer

Le harness dédié est `tests/bench_cache_sessions.py` :

```bash
# multi-session co-batch (le cas qui tombe à 0 %)
python3 tests/bench_cache_sessions.py \
  --sessions 4 --ctx-tokens 60000 --rounds 3 \
  --salt "$(date +%s)" --out results/cache-sessions.json

# sous la page 3584 (le cas que le fine-grained débloque)
python3 tests/bench_cache_sessions.py --sub-page --rounds 2

# préfixe partagé, queues différentes (fork de sous-agent)
python3 tests/bench_cache_sessions.py --shared-prefix --rounds 2
```

Le verdict `healthy` exige **round ≥ 2 hit ≥ 0,90** ; `collapsed` si ≤ 0,05.
Le round 1 est froid par construction. Utilisez un sel neuf à chaque run et
`GLM53_EXPOSE_CACHE_RESET=1` pour un reset propre.

## Portes d'adoption

1. **Per-group retention** : 2×68k et 4×60k ≥ 95 % ; replay 110k inchangé ;
   décode structured/prose dans le bruit ; pool identique. Si OK → garder on.
2. **Fine-grained APC** : prompt 2,6k hit ≥ 2 500 ; 6,5k hit ≥ 6 000 ;
   structured ≥ 97 % du contrôle ; pas d'IMA sur une longue génération
   (le clamp `KpoolTail` est déjà dans la lane). Si OK → passer à 1, sinon
   laisser 0.
3. **No-store** : une requête batch avec le flag ne doit pas changer le hit
   d'une session chaude voisine ; les deux receipts de suppression doivent
   apparaître dans le log.
4. **Métrique** : `vllm:prefix_cache_sparse_retention_misses` doit rester à 0
   quand tous les groupes participantes sont denses ; une valeur positive
   identifie un hit jeté.

## Rollback

| Levier | Rollback | Redémarrage |
|---|---|---|
| Rétention par groupe | `GLM53_APC_PER_GROUP_RETENTION=0` | oui |
| Fine-grained APC | `GLM53_FINE_GRAINED_APC=0` | oui |
| No-store | `GLM53_APC_NO_STORE=0` | oui |
| Métrique | `GLM53_PREFIX_CACHE_SPARSE_MISS_METRIC=0` | oui |

Aucun rebuild : les overlays sont montés à l'exécution et les knobs sont lus
au boot. Un rebuild n'est nécessaire que si l'image est gelée sans les scripts.

## Provenance

Porté de
[Reederey87/glm53-flash-exl3-2x-dgx-spark](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark)
(Apache-2.0), avec adaptations locales. Voir `vllm-exl3/NOTICE` et
`CREDITS.md`. Les tests hôtes correspondants sont dans
`vllm-exl3/tests/test_*apc*.py` et `test_prefix_cache_v2_wiring.py`.
