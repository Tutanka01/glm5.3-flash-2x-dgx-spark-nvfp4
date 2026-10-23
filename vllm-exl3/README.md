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

## À ne pas faire

- backend `--moe-backend marlin`, poids NVFP4, KV bf16 ou NVFP4 :
  `fp8_ds_mla` est la seule voie sparse-MLA sur SM12x ;
- épingler `TRITON_ATTN` pour le drafter : l'acceptation s'effondre
  ([QUALITY.md](../docs/exl3/QUALITY.md)) ;
- baisser `MAX_MODEL_LEN` pour « libérer » du KV, ou monter
  `MAX_NUM_BATCHED_TOKENS` pour « réparer » le prefix caching ;
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
| capacité long contexte à froid | `tests/bench_long_context.py` |
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
