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
| 2026-08-27 | `256k-graphs` | `--runs 3` | 9/9 | 0,32 s | 0,37 s | 14,4 tok/s | — | limite serveur 262k + petits prompts : capture bs=1 et décode court validés, **capacité 256k non testée** | `glm53-benchmark-20260827-181201.json` |
| 2026-08-28 | `256k` | long froid 240 000 | 1/1 | 204,59 s | — | 13,48 tok/s | préfill 1 173,12 tok/s | 240 008 tokens après template, 3/3 aiguilles, API saine ; eager, sans MTP, chunk 1024, statique 0,88 | `glm53-long-context-256k-safe-20260828-073956.json` |
| 2026-08-28 | `128k-dflash2` | `--runs 3` | 9/9 | 0,38 s | 0,42 s | 37,2 tok/s | — | DFlash2 C1 : ×2,6 vs sans spéculation, +29 % vs MTP5, TTFT intact ; concurrence et acceptation non encore mesurées | `glm53-benchmark-20260828-084231.json` |
| 2026-08-28 | `128k-dflash2` | `--runs 3 --concurrency 4` | 9/9 | 31,14 s | 83,79 s | 37,3 tok/s | 35,8 tok/s (132,6 s) | files sérielles : `MAX_NUM_SEQS=1`, pas une mesure batched ; décode par flux préservé, ~26 s d'attente par requête en file | `glm53-benchmark-20260828-090034.json` |
| 2026-08-28 | `128k-dflash2-c4` | `--runs 3` | 9/9 | 0,38 s | 0,42 s | 35,4 tok/s | — | C1 sur profil batché : −5 % vs profil mono, TTFT intact | `glm53-benchmark-20260828-092417.json` |
| 2026-08-28 | `128k-dflash2-c4` | `--runs 3 --concurrency 4` | 9/9 | 0,71 s | 1,02 s | 18,0 tok/s | 67,2 tok/s (70,3 s) | +61 % vs MTP5 batché, ×2,1 vs sans spéculation ; wall 70 s vs 145 s ; dépassé ensuite par le balayage c8 (86,0 à C6) | `glm53-benchmark-20260828-092536.json` |
| 2026-08-28 | `128k-dflash2-c8` | `--runs 3` | 9/9 | 0,38 s | 0,42 s | 37,2 tok/s | — | boot validé à statique 0,90 (0,92 : guard trip) ; C1 identique au profil mono | `glm53-benchmark-20260828-103119.json` |
| 2026-08-28 | `128k-dflash2-c8` | `--runs 3 --concurrency 2` | 9/9 | 0,53 s | 0,58 s | 26,3 tok/s | 50,5 tok/s (94,1 s) | palier intermédiaire du balayage | `glm53-benchmark-20260828-103254.json` |
| 2026-08-28 | `128k-dflash2-c8` | `--runs 3 --concurrency 4` | 9/9 | 0,72 s | 0,87 s | 20,0 tok/s | 64,9 tok/s (72,2 s) | cohérent avec le profil c4 ; p99 +13 % vs 0,77 s sans spéculation | `glm53-benchmark-20260828-103408.json` |
| 2026-08-28 | `128k-dflash2-c8` | `--runs 3 --concurrency 5` | 9/9 | 0,81 s | 0,86 s | 18,0 tok/s | 79,1 tok/s (60,2 s) | dépasse déjà le sommet C5 du port vLLM cité (56,2 tok/s) | `glm53-benchmark-20260828-103510.json` |
| 2026-08-28 | `128k-dflash2-c8` | `--runs 3 --concurrency 6` | 9/9 | 0,75 s | 0,82 s | 17,6 tok/s | 86,0 tok/s (55,6 s) | meilleur agrégé du cluster : ×2,06 vs MTP5 batché, ×2,7 vs sans spéculation ; pas de régression C6, wall ÷2,6 | `glm53-benchmark-20260828-103607.json` |
| 2026-08-28 | `128k-dflash2-c8` | `--runs 3 --concurrency 7` | 9/9 | 1,01 s | 1,02 s | 17,1 tok/s | 78,0 tok/s (60,9 s) | début de régression : agrégé −9 % vs C6, TTFT médian +34 % | `glm53-benchmark-20260828-104719.json` |
| 2026-08-28 | `128k-dflash2-c8` | `--runs 3 --concurrency 8` | 9/9 | 0,85 s | 1,21 s | 15,9 tok/s | 79,6 tok/s (60,0 s) | léger rebond d'agrégé mais p99 le plus haut du balayage ; sommet confirmé à C6 | `glm53-benchmark-20260828-104821.json` |

## Lecture des mesures du 2026-08-28

- **DFlash2 mono-flux : meilleur décode mesuré sur ce cluster.** 37,2 tok/s à
  128K, soit ×2,6 contre le même bench sans spéculation (14,5) et +29 % contre
  MTP5 (29,0). Le seuil de promotion « gain C1 ≥ 25 % contre MTP » de
  [DFLASH2.md](DFLASH2.md) est franchi dès le premier essai.
- **TTFT intact** : 0,38 s médian, p99 0,42 s. Le draft 1B ne pénalise pas
  l'admission en mono-flux, au contraire de MTP5 qui coûtait déjà +0,1 s.
- **Smoke chat + tools validé sur l'image dérivée** : parsing d'appel d'outil
  correct (`get_temperature`), génération déterministe conforme. Première
  validation fonctionnelle de la voie DFlash2, pas seulement un gain de débit.
- **Le run C4 sur `128k-dflash2` n'est pas une mesure batched** :
  `MAX_NUM_SEQS=1` place les requêtes excédentaires en file, d'où l'escalier de
  TTFT (médiane 31,1 s, p99 83,8 s, ~26 s d'attente par requête) et un agrégé
  (35,8 tok/s) quasi égal au mono. Aucun crash ni retract et un décode par flux
  préservé à 37,3 tok/s : la file se comporte proprement. La mesure de
  concurrence réelle attend `128k-dflash2-c4` puis le balayage c8.
- **DFlash2-c4 : meilleur des deux mondes, mesuré.** À concurrence 4, l'agrégé
  atteint 67,2 tok/s (+61 % vs MTP5 batché, ×2,1 vs sans spéculation) avec un
  TTFT médian de 0,71 s et un p99 de 1,02 s — MTP5 s'effondrait à 7,6 s/45,3 s
  sur le même test, et le wall clock passe de 145 s à 70 s. Le décode par flux
  partagé retombe à ~18 tok/s ; le dernier run dégagé remonte à 34,9 tok/s dès
  que le batch se vide. Réserves du protocole : le p99 C4 reste 33 % au-dessus
  de la référence sans spéculation (0,77 → 1,02 s), au-delà du garde strict de
  10 % bien qu'excellent en absolu, et le gain C1 du profil batché est de +22 %
  (35,4 tok/s) contre +29 % sur le profil mono.
- **Balayage c8 à statique 0,90 : sommet à C6, 86,0 tok/s agrégés.** Le profil
  a booté après dérivation et a enchaîné C1→C8 sans incident : agrégé 50,5
  (C2) → 64,9 (C4) → 79,1 (C5) → **86,0 (C6, wall 55,6 s)**, puis régression
  78,0 (C7) et 79,6 (C8) avec un p99 qui franchit 1 s. Même forme de courbe
  que le port vLLM (sommet puis déclin), décalée d'un cran : C6 au lieu de C5,
  et un sommet +53 %. Le point d'exploitation recommandé est donc C6 ; au-delà,
  seule la latence se dégrade, pas la stabilité. À C6, TTFT médian 0,75 s et
  p99 0,82 s, soit +7 % vs la référence sans spéculation — dans le garde de
  10 % du protocole ; à C4 le p99 est à +13 %, juste au-dessus. Contre MTP5
  batché, l'agrégé est ×2,06 et le p99 TTFT ÷55.
- **Rapprochement avec le port vLLM cité** : 46,9 tok/s C1 chaud sur code pur
  contre 37,2 ici sur le mix standard (sanity/coding/reasoning, température 0).
  Cohérent : le mix contient du raisonnement et de la prose, moins acceptés que
  le code.
- **Portes qualité : une franchie, deux partielles.** (1) *Acceptation* : les
  logs de décode à C6 affichent `accept len` 4,38–5,06 pour 8 tokens draftés
  (taux 0,48–0,58) — au-dessus du seuil mix ≥ 35 %, médiane ~0,5 autour du
  seuil code ≥ 50 %. (2) *Égalité des sorties température 0* (hash vs la
  référence sans spéculation du 27/08) : `coding` 3/3 **identique** octet pour
  octet ; `sanity` et `reasoning` diffèrent. Le profil est compatible avec des
  quasi-égalités de logits tranchées différemment entre images (kernels FA4 de
  l'image dérivée vs image de base), pas avec un échec de vérification — la
  classe code, cible principale de DFlash2, est la plus longue et tombe
  exactement juste. À confirmer par une comparaison élargie avant promotion.
  (3) *Marge mémoire* : `MemAvailable` head mesuré à 7,3 GiB pendant un bench
  C6 — au-dessus du plancher du garde (6 GiB) mais sous le critère de
  promotion (8 GiB) ; marge worker non relevée. Un statique 0,88 rendrait
  ~9,7 GiB si le c8 devenait un profil de production, au prix du pool KV.
  Le « pire prefill » réel (prefills concurrents longs) reste à mesurer.
- **Non mesuré à ce stade** : élucidation de l'écart de hash
  sanity/reasoning (numérique inter-images vs défaut de vérification), marge
  mémoire ≥ 8 GiB (7,3 GiB mesurés à C6 sur le head), acceptation par classe
  stricte et stabilité longue. La promotion hors `experimental` attend ces
  points (protocole dans [DFLASH2.md](DFLASH2.md)).
- **Première tentative `128k-dflash2-c8` : arrêt au démarrage à 09:43, cause
  identifiée.** Ce n'est ni un bug DFlash2 ni un crash CUDA : le garde mémoire
  de démarrage du head a coupé le conteneur lui-même (`GUARD TRIP`,
  `MemAvailable` 6032 MiB < plancher 6144 MiB ; `OOMKilled=false`, exit 137)
  pendant la capture des graphes draft bs=8, alors qu'il ne restait que 5,81 Go
  de marge GPU côté head à statique 0,92 (poids 91,0 Go + draft 1,65 Go + Mamba
  40 slots ~4,1 Go + pools KV ~4,8 Go). Les warnings `NVRM NV_ERR_NO_MEMORY`
  du driver corroborent l'épuisement de la mémoire unifiée ; le worker, mieux
  doté (7,98 Go à la même étape), a survécu et n'a perdu le rang 0 qu'en
  surface (`Broken pipe` TCPStore). Le garde a fait exactement son travail :
  la veille au soir, un OOM kernel sur ce même head avait tué le scheduler
  **et** `systemd`. Le profil est dérivé à statique 0,90 (~+2,4 Go de marge,
  réserve identique au c4 validé) ; repli 0,88 si nouveau trip, après
  `collect-glm53-report.sh`. Les rangs n'ont jamais atteint la readiness :
  aucune conclusion sur DFlash2 en charge.

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
- **Le run `256k-graphs` n'est pas un test 256k** : ses prompts font quelques
  dizaines de tokens. Il prouve que la capture et le replay court passent avec
  `context-length=262144`, pas qu'un préfill froid proche de cette limite
  produit un premier token. Cette capacité reste « non mesurée » tant que le
  protocole ci-dessous n'a pas réussi.

## Matrice de capacité long-contexte

Le checkpoint est nativement configuré pour 1 048 576 tokens, mais la fenêtre
réellement utilisable dépend du pool KV, des workspaces et du chemin de replay.
Ne cochez une ligne qu'après un prompt **froid** réellement envoyé :

| Profil | Cible froide | Graphes | Spéculation | KV | Préfill CP | État |
|---|---:|---|---|---|---|---|
| `256k-mtp` | 240 000 | oui | 5 | FP8 | non | **échec** : préfill figé puis scheduler `-9`; quarantaine |
| `256k` | 240 000 | non | non | FP8 | non | **réussi le 2026-08-28** : 240 008 tokens, récupération 3/3, API saine |
| `256k-dflash2-eager` | 240 000 | non | DFlash2 1B | FP8 | non | à mesurer ; Mamba BF16/5, fenêtre draft 2048, statique 0,88 |
| `384k-quality` | 360 000 | oui | 5 | BF16 | 2 rangs | à mesurer |
| `512k-mtp-eager` | 480 000 | non | 5 | FP8 | non | à mesurer ; évite le bug graph mais reste non sûr côté mémoire |
| `512k-mtp-cp` | 480 000 | oui | 5 | FP8 | 2 rangs | à mesurer ; expérimental |

Commande canonique :

```bash
./bench-long-context.py \
  --target-tokens 240000 \
  --cold \
  --label 256k-safe
```

Cette commande suppose que `./start-glm53.sh 256k` a lancé le profil réellement
actif. `--label` ne change pas le profil. Le client calibre le texte avec l'endpoint tokenizer du serveur, place trois
aiguilles vers 5 %, 50 % et 95 %, mesure le TTFT/préfill et le décode, puis
interroge `/v1/models` après la réponse. La ligne ne passe que si les trois
codes sont retrouvés et si l'API est encore saine. Au-dessus de 128K, le profil
de fiabilité est contrôlé automatiquement. Les voies MTP, graphes, CP ou
DFlash2 restent lançables avec `--allow-unsafe-profile`, qui consigne le
contournement dans le JSON.

Pourquoi ce protocole est bloquant : [SGLang #36550](https://github.com/sgl-project/sglang/issues/36550)
rapporte un crash de replay CUDA graph au premier token après un préfill froid
supérieur à 262 144 tokens. Le context parallelism de préfill à deux rangs a
fait passer 428k dans le reproducer amont, mais il ne constitue pas une preuve
sur GB10/FlashInfer tant que la même requête n'a pas réussi ici.

Le succès `256k` tranche aussi la question du chunk sur cette recette : 1024 a
fonctionné avec SGLang à 240 008 tokens. Le segfault sous 2048 rapporté par le
port DFlash2 concernait sa voie vLLM patchée ; ce n'est pas un plancher général
du modèle et il ne doit pas remplacer la preuve obtenue sur le runtime présent.
