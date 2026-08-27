# Historique des benchmarks

Mesures de débit relevées sur cluster réel (2× GB10, TP=2, RoCE). Les artefacts
JSON bruts restent dans `results/` (ignoré par Git) ; ce tableau en conserve
l'essentiel pour suivre les régressions et les gains de chaque levier.

Pour contribuer une ligne : `./bench-glm53.py --runs 3 --concurrency N`, puis
recopiez le résumé médian dans le tableau en précisant le profil et la date.

| Date | Profil | Bench | Succès | TTFT méd. | TTFT p99 | Décode méd. | Agrégé | Notes | Artefact |
|---|---|---|---|---:|---:|---:|---:|---|---|
| 2026-08-27 | `128k-batch4` | `--runs 3 --concurrency 4` | 9/9 | 0,55 s | 0,77 s | 11,7 tok/s | 31,5 tok/s (144,9 s) | sans MTP ; première mesure batched | `glm53-benchmark-20260827-130517.json` |

## Lecture de la mesure du 2026-08-27

- **Scaling de la concurrence** : 31,5 tok/s agrégés pour 11,7 tok/s par
  requête, soit 2,7× à concurrence 4. La mise en lot exploite bien la bande
  passante mémoire autrement idle en mono-flux.
- **TTFT** : 0,44-0,77 s mesuré sur des prompts courts ; le préfill chunked
  (4096 tokens) et le radix cache absorbent les redémarrages de sous-agents.
- **Interférence de lot attendue** : le run `reasoning 3/3` est descendu à
  7,6 tok/s contre 11,8 pour ses jumeaux (température 0, mêmes tokens) car il
  chevauchait les décodes `coding`. C'est le comportement normal d'un scheduler
  batched : le p99 par requête se dégrade, le goodput total monte.
- **Attendu du MTP** : à 85 ms/token, l'étape de décode laisse le GPU largement
  inactif ; la spéculation NEXTN (profil `128k-batch4-mtp`) devrait lever le
  débit mono-flux de façon nette, avec un gain réduit à concurrence 4.
