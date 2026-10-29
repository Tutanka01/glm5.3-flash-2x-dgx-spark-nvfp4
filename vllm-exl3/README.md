# Lane vLLM / EXL3

[← Accueil du dépôt](../README.md) · Autre lane : [SGLang / NVFP4](../sglang-nvfp4/README.md)

GLM-5.3-Flash en poids **EXL3/TR3 4bpw** servi par **vLLM** sur 2× DGX Spark
(TP=2 sur CX7/RoCE). Le code provient de la recette
[MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
(MIT, commit `e25aa70`), vendorée puis durcie dans ce dépôt.
**Toutes les commandes de ce guide se lancent depuis `vllm-exl3/` sur le head.**

> **État :** validée sur le cluster le 2026-08-29 : **990 007 tokens froids**
> retrouvés au needle-exact (fenêtre 1M complète), 66,7 tok/s en décode
> structured (acceptation 0,959), prefix-cache 4,1× sur les follow-ups, poids
> ≈ FP8 officiel (KLD 0,0246 contre 0,0605 pour le NVFP4). Elle n'est pas
> encore la lane par défaut tant que sa
> [checklist de promotion](../docs/BENCHMARKS.md#checklist-de-promotion-exl3)
> n'est pas close. Tous les chiffres et artefacts sont dans
> [docs/BENCHMARKS.md](../docs/BENCHMARKS.md).

La référence technique complète, en anglais (overlay, KV, prefix caching,
toutes les variables de `.env`, kit TP=4), est dans [REFERENCE.md](REFERENCE.md).

## Prérequis

- 2× DGX Spark / ThinkStation PGX GB10, SSH sans mot de passe head → worker ;
- Docker utilisable sans `sudo` sur les deux nœuds ;
- environ **180 GiB libres par nœud** (checkpoint ~164 GiB) ;
- la lane SGLang **arrêtée** : les deux lanes ne partagent aucun état mais se
  disputent la mémoire unifiée des GB10.

## Démarrer

```bash
cd vllm-exl3
cp .env.example .env        # éditer HEAD_IP / WORKER_IP / NIC / GID
./download.sh               # optionnel : ~164 GiB dans le cache HF du head
./start.sh                  # tire l'image GHCR, synchronise le worker, lance TP=2
```

Lisez le bloc RoCE de `.env.example` avant le premier démarrage : noms de NIC,
indices GID par rang et repli mémoire y sont propres à chaque kit.

- API : `http://<HEAD_IP>:8888/v1`, modèle `GLM-5.3-Flash-EXL3`.
- L'export du shell l'emporte sur `.env` :
  `SPEC_METHOD=mtp ./start.sh restart` repasse la spéculation en MTP k=2.
- Pour les petits budgets de tokens, envoyez
  `chat_template_kwargs: {"enable_thinking": false}`.
- Agents concurrents : posez `GLM53_MIXED_PREFILL_CHUNK=256` dans `.env`. Le
  script démarre en `skip`, mais l'A/B C4 du 2026-08-29 a tranché pour `256`
  (p99 TTFT 14,8 → 1,2 s, agrégé +29 %).
- Arrêt : `./stop.sh`.

> **Premier démarrage après le sync du 2026-09-02.** L'image GHCR publique
> `:exl3` date du 2026-08-28. Le premier `./start.sh` détecte l'écart de
> recipe stamp et **reconstruit l'image localement**, couche CUDA comprise :
> comptez longtemps. `SKIP_BUILD=1` reste sur l'image GHCR, avec
> `EXL3_FAT_KERNEL=0`.

Ce sync amont apporte le kernel E2 fat-expert (**+20–21 % de prefill froid**
de 8k à 300k, décode inchangé), `MAX_NUM_BATCHED_TOKENS=7168`,
`DFLASH_DRAFT_TP=2`, le garde-fou de config numérique et
`GLM53_INDEXER_WORKSPACE=rightsize` en opt-in (~+26 % de KV). Notre correctif
local #31 (endpoint de reset du prefix cache) est conservé.

### Contexte maximal : cache KV insuffisant au démarrage

Si vLLM indique `To serve at least one request with the model's max seq len`,
la longueur demandée dépasse la capacité du cache KV disponible. Par exemple,
768 000 tokens nécessitant 12,88 GiB avec seulement 12,56 GiB disponibles font
échouer le démarrage ; vLLM estime alors un plafond à 723 968 tokens pour
ce budget mémoire. Ce plafond dépend des réservations du lancement.

Pour viser le contexte de 1M de la recette, vérifiez d'abord l'espace de travail
de l'indexer : le lanceur démarre par défaut en `stock`, qui réserve environ
4,92 GiB à 1M. Avec `MAX_NUM_SEQS=4`, `rightsize` ramène cet espace à environ
126 MiB sans réduire la longueur de contexte. La référence rapporte un
[démarrage à 1M avec ce mode](REFERENCE.md#kv-cache-this-kit-2026-08-29).
Pour activer ce correctif existant sur les deux rangs :

```bash
MAX_MODEL_LEN=1000000 GLM53_INDEXER_WORKSPACE=rightsize ./start.sh restart
curl --noproxy '*' -fsS http://127.0.0.1:8888/health
```

Ces valeurs passées au lancement l'emportent sur `.env`. Pour les conserver,
réglez `MAX_MODEL_LEN=1000000` et `GLM53_INDEXER_WORKSPACE=rightsize` dans votre
`.env`. Le résultat reste à mesurer sur votre kit : vérifiez les lignes
`[glm53-indexer-workspace] rightsize:` / `builder:` sur les deux rangs, puis
`Available KV cache memory`, `GPU KV cache size` et `Maximum concurrency`.
Si `rightsize` était déjà actif, comparez les poids, activations, réservations
CUDA graphs et `GPU_MEM_UTIL` avant d'attribuer le manque au workspace.
Ne désactivez pas l'estimation des CUDA graphs sans mesurer leur coût réel.
Adaptez l'URL si `PORT` a changé.
Les erreurs worker `Broken pipe` qui suivent l'arrêt du head sont secondaires.
Une réponse HTML sur `/v1/models` ne suffit pas à diagnostiquer le routage :
le contrôle local ci-dessus permet de vérifier le serveur sans proxy.

## Variante abliterated (`ABLIT=1`)

La lane peut servir GLM-5.3-Flash sans refus de sécurité :

```bash
ABLIT=1 ./start.sh restart     # ou ABLIT=1 dans .env
```

Au premier démarrage, le head télécharge environ 2,6 GiB de tenseurs `o_proj`
du donneur épinglé (Dealign, commit `745aac2f`) et les copie sur le worker.
Les deux rangs les substituent ensuite au chargement : aucun poids n'est
réécrit sur disque et rien n'est reconstruit. `ABLIT=0` revient au modèle
d'origine au redémarrage suivant.

- Recette « altered Dealign » : couches 15–43 et bloc MTP 45 modifiés ;
  couches 0–14 **et 44** laissées d'origine.
- Fail-closed : révision du donneur épinglée, sha256 vérifié sur chaque rang,
  aucun repli silencieux sur une méthode inefficace.
- Modèle servi sous l'id `GLM-5.3-Flash-EXL3-Ablit` ; l'id d'origine reste
  accepté en alias pour les clients existants.
- Les mesures de qualité ont été faites **sans** ablit : relancez
  `tests/bench_decode.py` après activation. Gardez la clé API et le filtrage
  réseau : le modèle répond à tout ce que le client envoie.

Détails et réglages `ABLIT_*` : [REFERENCE.md](REFERENCE.md#abliteration-ablit1).

## Performance agentique (opt-in, 2026-10)

Trois briques sont ajoutées derrière des knobs, sans changer le chemin par
défaut tant qu'elles n'ont pas leur A/B sur ce kit.

**Prefix cache multi-session.** Le cache hybride KDA+MLA est aligné sur des
pages de 3584 tokens ; en multi-session, un groupe qui manque son checkpoint
peut annuler tous les hits (0 % silencieux). Le bundle
`overlay/patch_apc_per_group_retention.py` +
`patch_fine_grained_apc.py` + `patch_apc_no_store.py` +
`patch_prefix_cache_sparse_miss_metric.py` ferme ces trous :

```bash
# rétention par groupe : MLA/mamba dense, drafter SWA aux frontières (défaut on)
GLM53_APC_PER_GROUP_RETENTION=1
# hits fins à 64 tokens : gros gain sous la page 3584, coût ~-2,7 % structured
GLM53_FINE_GRAINED_APC=0        # passer à 1 après l'A/B
# opt-out d'écriture par requête (batch/eval) ; read-hit conservé
GLM53_APC_NO_STORE=1
# métrique vllm:prefix_cache_sparse_retention_misses
GLM53_PREFIX_CACHE_SPARSE_MISS_METRIC=1
```

Protocole et chiffres attendus : [docs/exl3/PREFIX-CACHE.md](../docs/exl3/PREFIX-CACHE.md).
Mesure : `tests/bench_cache_sessions.py` (2×68k, 4×60k, sous-page, préfixe
partagé).

**Fat GEMM 3-stage cp.async.** `EXL3_FAT_PIPELINE=1` sélectionne
`exllamav3_ext.exl3_fat_gemm_pipe` (Reederey87 S2b, Apache-2.0) : même ABI,
même sémantique, pipeline `cp.async` 3 étages. Référence : +38–41 % de débit
noyau à M=3584, bit-exact. Défaut `0` ; activer après le ladder froid 8k/100k/
300k. Le premier démarrage après le changement de Dockerfile rebuilde l'image
(recipe stamp), comptez le temps de compilation CUDA. Détails :
[docs/DESIGN-grouped-fat-moe.md](docs/DESIGN-grouped-fat-moe.md).

**Warmup préfill.** `GLM53_WARMUP_PREFILL_TOKENS=8192` brûle un préfill 8k
après `/health`, puis reset le prefix cache. Le premier prompt long n'est plus
le premier JIT. Défaut `0` (boot plus court).

**Adaptive-K de vérification.** `GLM53_ADAPTIVE_K=ema` (défaut `off`) trime
uniquement `request.spec_token_ids` — jamais l'exécution du drafter DFlash2 —
et ajoute les graphes cibles q=3/5. Référence : +7–11 % prose/essay à C4, coût
~1,1–1,2 GiB/rank. Protocole A→B0→B et rollback :
[docs/DESIGN-adaptive-k.md](docs/DESIGN-adaptive-k.md).

**Prochains chantiers documentés :** dispatch groupé des fat experts
([docs/DESIGN-grouped-fat-moe.md](docs/DESIGN-grouped-fat-moe.md)).
La variante `cp.async` est déjà disponible (`EXL3_FAT_PIPELINE=1`), default
off et bit-exact sur les formes testées.

### Qualifier un changement

```bash
# A/B complet : restart par bras, health gate, warmup jeté, reset du cache,
# médianes + p99, artefacts JSON et tableau Markdown.
python3 scripts/bench-ab.py \
  --arm baseline \
  --arm candidate:MAX_NUM_SEQS=6,GLM53_MIXED_PREFILL_CHUNK=448 \
  --workdir . --runs 3 \
  --out results/ab-$(date -u +%Y%m%d-%H%M%S).json

# Cache multi-session seul (2×68k, 4×60k, sous-page, préfixe partagé).
python3 tests/bench_cache_sessions.py \
  --sessions 4 --ctx-tokens 60000 --rounds 3 --salt "$(date +%s)"

# Détecteur de préfill dégradé (lecture seule, jamais de restart).
python3 scripts/prefill-watchdog.py --json --window 3 --threshold 0.30
```

Le protocole et les portes sont dans
[docs/exl3/PREFIX-CACHE.md](../docs/exl3/PREFIX-CACHE.md) et
[docs/BENCHMARKS.md](../docs/BENCHMARKS.md#nouveaux-leviers-2026-10--protocole-et-portes).

## À ne pas faire

- backend `--moe-backend marlin`, poids NVFP4, KV bf16 ou NVFP4 :
  `fp8_ds_mla` est la seule voie sparse-MLA sur SM12x ;
- épingler `TRITON_ATTN` pour le drafter : l'acceptation s'effondre
  ([QUALITY.md](../docs/exl3/QUALITY.md)) ;
- baisser `MAX_MODEL_LEN` pour « libérer » du KV, ou monter
  `MAX_NUM_BATCHED_TOKENS` pour « réparer » le prefix caching ; en cas d'échec
  mémoire au démarrage, vérifiez les réservations décrites ci-dessus ;
- détruire les caches HF, requantizer, faire un `docker rm` des caches ;
- changer TP, les pins CX7 ou `USE_HOST_NCCL` sans replomber NCCL ;
- combiner `tools` avec `response_format`/`guided_json` dans une même requête
  (interdiction amont).

Attendez-vous aussi à des stalls de prefill (10–40 s) quand de gros contextes
arrivent en même temps : lisez
[KNOWN-ISSUES.md](../docs/exl3/KNOWN-ISSUES.md) avant toute charge agentique.

## Outils de la lane

| Besoin | Commande (depuis `vllm-exl3/`) |
|---|---|
| décode structured / prose | `tests/bench_decode.py` |
| prefix cache froid / chaud | `tests/bench_prefix_cache.py` |
| cache multi-session (2×68k, 4×60k, sous-page) | `tests/bench_cache_sessions.py` |
| capacité long contexte à froid | `tests/bench_long_context.py` |
| A/B automatisé (restart, health, warmup jeté, reset, médianes/p99) | `scripts/bench-ab.py` |
| détecteur de préfill dégradé | `scripts/prefill-watchdog.py` |
| bit-exact du fat-GEMM `cp.async` | `tests/bench_fat_gemm_pipe.py` |
| bench court et concurrence (commun aux lanes) | `../tools/bench-glm53.py --model GLM-5.3-Flash-EXL3` |
| soak tool-calling (issue #10) | `tests/soak_tool_calls.py` |
| A/B scheduler C4 | `scripts/c4-chunk-ab.sh` + `tests/compare_c4.py` |
| A/B qualité contre l'API officielle | `tests/grade_ab_quality.py` + `tests/ab_quality_prompts.jsonl` |
| sonde de soak quotidien | `scripts/soak-day.sh` ([SOAK.md](../docs/exl3/SOAK.md)) |
| tests sans GPU | `tests/run-local.sh` |

## Documentation

- [REFERENCE.md](REFERENCE.md) : référence technique détaillée (EN) ;
- [QUALITY.md](../docs/exl3/QUALITY.md) : KLD, acceptation DFlash2 ;
- [KNOWN-ISSUES.md](../docs/exl3/KNOWN-ISSUES.md) : issues amont connues ;
- [SOAK.md](../docs/exl3/SOAK.md) : protocole et journal du soak multi-jours ;
- [PREFIX-CACHE.md](../docs/exl3/PREFIX-CACHE.md) : cache agentique
  multi-session, knobs et portes d'adoption ;
- [DESIGN-adaptive-k.md](docs/DESIGN-adaptive-k.md) : adaptive-K de
  vérification (default off) ;
- [DESIGN-grouped-fat-moe.md](docs/DESIGN-grouped-fat-moe.md) : fat-GEMM
  `cp.async` et dispatch groupé ;
- [docs/](docs/) : notes de conception amont (prefill froid, workspace de
  l'indexer) ;
- [BENCHMARKS.md](../docs/BENCHMARKS.md) : journal commun aux deux lanes.

## Licence et sources

Le code vendoré est sous MIT ([LICENSE](LICENSE)). L'amont est passé en
AGPL-3.0 le 2026-09-07 : aucun code postérieur n'entre ici sans décision de
licence explicite, et `scripts/check-vendor-license.sh <ref>` refuse toute
ref non MIT. Politique complète : [CREDITS.md](../CREDITS.md#politique-de-licence-du-vendor-vllm-exl3).

- [recette EXL3 vLLM de MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) ;
- [checkpoint EXL3/TR3 4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
  et [son miroir public](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw) ;
- [drafter DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) (CC BY-NC-ND 4.0) ;
- [kernels ExLlamaV3](https://github.com/turboderp-org/exllamav3) (format EXL3).
