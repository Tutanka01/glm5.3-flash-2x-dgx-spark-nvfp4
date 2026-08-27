# Historique des benchmarks

Mesures de débit relevées sur cluster réel (2× GB10, TP=2, RoCE). Les artefacts
JSON bruts restent dans `results/` (ignoré par Git) ; ce tableau en conserve
l'essentiel pour suivre les régressions et les gains de chaque levier.

Pour contribuer une ligne : `./bench-glm53.py --runs 3 --concurrency N`, puis
recopiez le résumé médian dans le tableau en précisant le profil et la date.

| Date | Profil | Bench | Succès | TTFT méd. | TTFT p99 | Décode méd. | Agrégé | Notes | Artefact |
|---|---|---|---|---:|---:|---:|---:|---|---|
| 2026-08-27 | `128k-batch4` | `--runs 3 --concurrency 4` | 9/9 | 0,55 s | 0,77 s | 11,7 tok/s | 31,5 tok/s (144,9 s) | sans MTP ; première mesure batched | `glm53-benchmark-20260827-130517.json` |
| 2026-08-27 | `128k-batch4` | `--runs 3` | 9/9 | 0,31 s | 0,36 s | 14,5 tok/s | — | référence mono-flux sans MTP | `glm53-benchmark-20260827-132411.json` |
| 2026-08-27 | `32k-mtp` | `--runs 3` | 9/9 | 0,39 s | 0,56 s | 29,0 tok/s | — | MTP : ×2,0 vs mono-flux sans MTP | `glm53-benchmark-20260827-134305.json` |
| 2026-08-27 | `128k-batch4-mtp` | `--runs 3` | 9/9 | 0,40 s | 0,53 s | 28,9 tok/s | — | décode identique à 32k-mtp : le long contexte ne pénalise pas le décode | `glm53-benchmark-20260827-140200.json` |
| 2026-08-27 | `128k-batch4-mtp` | `--runs 3 --concurrency 4` | 9/9 | 7,61 s | 45,3 s | 21,7 tok/s | 41,8 tok/s (114,7 s) | agrégé +33 % mais TTFT dégradé par l'admission retardée sous spéculation | `glm53-benchmark-20260827-140538.json` |

## Lecture des mesures du 2026-08-27

- **Scaling de la concurrence sans MTP** : 31,5 tok/s agrégés pour 14,5 tok/s
  mono-flux (2,2× à concurrence 4, compte tenu du débit mono plus élevé). La
  mise en lot exploite la bande passante mémoire autrement idle.
- **TTFT sans MTP** : 0,31 s mono, 0,77 s p99 en batché — le préfill chunked
  (4096 tokens) et le radix cache absorbent les rafales de sous-agents.
- **Interférence de lot attendue** : le run `reasoning 3/3` est descendu à
  7,6 tok/s contre 11,8 pour ses jumeaux (température 0, mêmes tokens) car il
  chevauchait les décodes `coding`. Comportement normal d'un scheduler batched :
  le p99 par requête se dégrade, le goodput total monte.
- **MTP mono-flux : ×2,0 confirmé** (14,5 → 29,0 tok/s), TTFT quasi inchangé
  (+0,1 s, coût du draft). Identique à 32k et 131k de contexte.
- **MTP batché : compromis défavorable à la latence**. Agrégé +33 %
  (31,5 → 41,8 tok/s) mais TTFT médian 7,6 s et p99 45,3 s : chaque étape de
  décode spéculatif est plus lourde et l'admission des nouveaux prefills attend
  les frontières de batch.
- **Recommandation issue des mesures** : `128k-batch4-mtp` pour l'usage
  interactif mono-flux, `128k-batch4` sans MTP pour les rafales de sous-agents.
  Pistes intermédiaires : MTP à 3 étapes, ou MTP à concurrence 2.
