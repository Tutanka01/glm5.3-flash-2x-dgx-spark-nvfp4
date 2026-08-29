# Historique des benchmarks — source unique des deux lanes

Ce fichier est **la seule source de vérité** pour les mesures du cluster
(2× GB10, TP=2, RoCE) : la lane SGLang/NVFP4 (défaut) et la lane EXL3/vLLM
(branche `dev`). Aucun autre fichier du dépôt ne doit porter des chiffres de
bench : les docs de lane renvoient ici.

Les artefacts JSON bruts restent dans `results/` (racine) ou `vllm-exl3/results/`
(lane EXL3) — ignorés par Git ; ce journal en conserve l'essentiel avec le nom
du fichier pour suivre régressions et gains.

Règles communes aux deux lanes (culture benchmark du repo + kit vendoré) :

- noter le protocole avec chaque nombre : classe de prompt, runs, tokens,
  température, thinking on/off ;
- les nombres upstream notent **leur kit** : ce sont les baselines à battre ou
  reproduire, pas un claim sur le nôtre ;
- une ligne pour un kit n'est valide qu'avec son artefact JSON (garder le nom
  du fichier dans la ligne) ;
- ne jamais citer un tok/s structured sans le tok/s prose à côté (régimes
  d'acceptation ~2,8×, voir [EXL3-QUALITY.md](EXL3-QUALITY.md)).

Pour contribuer une ligne : `./bench-glm53.py --runs 3 --concurrency N`, puis
recopiez le résumé médian dans le tableau de la lane concernée en précisant le
profil et la date.

## Lane SGLang / NVFP4 (lane par défaut)

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
| 2026-08-28 | `256k-dflash2-eager` | long froid 180 000 | 1/1 | 148,27 s | — | 40,47 tok/s | préfill 1 214,06 tok/s | 180 005 tokens après template, 3/3 aiguilles, API saine ; DFlash2 ×3,0 vs `256k` sans spéculation (13,48) ; chunk 2048, statique 0,88, mamba usage 0,80 au pire | `glm53-long-context-256k-dflash2-180000-20260828-134329.json` |
| 2026-08-28 | `256k-dflash2-eager` | long froid 220 000 | 0/1 | — | — | — | — | échec propre : stream fermé sans aucun token (hash de chaîne vide), API saine après ; confirmé par les logs : **aucun prefill lancé**, refus d'admission — pool ≈ 210K tokens < 220K demandés | `glm53-long-context-256k-dflash2-220000-20260828-134711.json` |
| 2026-08-28 | `256k-dflash2-eager` | long froid 200 000 | 1/1 | 162,56 s | — | 39,61 tok/s | préfill 1 230,42 tok/s | 200 012 tokens après template, 3/3 aiguilles, API saine ; usage pool ~0,95 — plafond pratique de la lane à statique 0,88 | `glm53-long-context-256k-dflash2-200000-20260828-135605.json` |

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
| `256k-dflash2-eager` | 240 000 | non | DFlash2 1B | FP8 | non | **180 000 et 200 000 réussis le 2026-08-28** (3/3 aiguilles, API saine, décode ~40 tok/s) ; **220 000 refusé à l'admission** (aucun prefill loggé, pool ≈ 210K tokens) — dépasser ~210K exige un statique supérieur, essai explicite `--allow-unsafe-profile` |
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

## Lane EXL3 / vLLM (branche `dev`) — 2× GB10, TP=2, RoCE, ThinkStation PGX, 2026-08-29

Chemin de service = celui d'amont, à l'octet près (image tirée, aucune
modification de source) ; durcissement côté hôte uniquement. Données terrain
aussi remontées en amont : issue MiaAI-Lab #32. Les protocoles se lancent
depuis `vllm-exl3/` ; les artefacts vivent dans `vllm-exl3/results/`.

### Baseline amont (kit MiaAI-Lab, 2026-08-28, DFlash2 k=7, temp 0, thinking off, 400 tok, CUDA graphs)

Décode, protocole sparkDash — Structured = count 1→200 (forte acceptation),
Code = prompts clamp. Le tok/s stream est par requête ; l'agrégé compte tous
les flux.

| Concurrence | TTFT | Stream tok/s | Agrégé tok/s |
|---:|---:|---:|---:|
| ×1 | 719 ms | 62.9 (structured) | 62.9 |
| ×2 | 6,62 s | 51.7 | 103.3 |
| ×4 | 6,30 s | 37.1 | 146.5 |

`tests/bench_decode.py` du lab, même protocole, médiane de 5 × 400, C1 :
structured **61.7** tok/s (accept 0.918 / 6.43 par pas) ; prose **26.9**
(0.332 / 2.33). Long contexte / mixte (~60–100k KV) 24–27. Baseline MTP k=2
~24.6.

Prefix caching (service 1M, vrai user + assistant + follow-up) :

| Tour | Hits | Prompt tok | TTFT |
|---|---:|---:|---:|
| ~7,7k froid | 0 | 7696 | 9,7 s |
| ~7,7k follow-up | 7168 (93 %) | 7717 | **1,17 s** |
| ~12k follow-up | 10752 | 12015 | 1,94 s |
| ~16k follow-up | 14336 | 16015 | 2,18 s |
| 4× ~7,5k follow-ups concurrents | 7168 chacun | 7515 chacun | 1,86–2,50 s |

Capacité de contexte : `max_model_len` 1M avec pool 1 754 237 tokens (1,75×) à
util 0.87 ; ~256k ×3 concurrents tenus en live (29,5 % de KV au pic).

### Ce kit (2026-08-29)

| Date | Protocole | Résultat | Notes | Artefact |
|---|---|---|---|---|
| 2026-08-29 | `bench_decode.py --phase structured --structured --runs 5 --max-tokens 400` | **66.7 tok/s** médian (63.3–68.6), TTFT 0.46 s, accept 0.959 / 6.71 par pas | **au-dessus de la baseline amont** (61.7 lab / 62.9 sparkDash, accept 0.918) ; par position 1.0/1.0/1.0/0.98/0.97/0.90/0.89 — pas d'effondrement tardif, chemin d'attention du drafter sain. Chemin de service identique : l'écart vient de la variance kit + de l'échantillonnage probabiliste du draft, pas de nos changements | `/tmp/exl3-structured.json` |
| 2026-08-29 | `bench_decode.py --phase prose --runs 5 --max-tokens 400` | 25.2 tok/s médian (23.6–26.9), TTFT 0.46 s, accept 0.305 / 2.13 | dans la bande amont (26.9 lab / 27.1 second kit) ; la forme par position colle à la signature DFlash2 — l'asymétrie ~2.6× structured/prose est le caractère connu du drafter | `/tmp/exl3-prose.json` |
| 2026-08-29 | `bench_prefix_cache.py` v2.1+ (`--prompt-tokens 8400`) | **hit 0.8541, eff 0.9999** — chaque page 3584 complète du prompt chaud réutilisée ; TTFT chaud 2.5 s vs 10.3 s vrai froid (**4.1×**) | modèle de pages confirmé : les hits sont alignés sur la **page hybride MLA de 3584 tokens** (`floor(tokens/3584)×3584`), et 7168 = 2×3584 exactement. Les bizarreries v1/v2.0 venaient de l'instrumentation du bench, pas de la stack : une fenêtre métriques couvrant froid+chaud divise le ratio par deux, et la réutilisation de contenu entre sessions fabrique de faux froids car l'image n'a **pas d'endpoint de reset du cache** (documenté en amont, issue #31 ; corrigé côté client par le sel de session) | `vllm-exl3/results/glm53-exl3-prefix-cache-*.json` |
| 2026-08-29 | `./bench-glm53.py --runs 3 --concurrency 4 --thinking off` | 9/9 ok, agrégé **54.8 tok/s**, TTFT médian 1.61 s, **p99 16.7 s** | escalier de TTFT conforme à `GLM53_MIXED_PREFILL_CHUNK=skip` : les nouveaux prefills attendent qu'un décode se vide (coding r2 TTFT 9.54 s ≈ total coding r1 9.50 s). Tradeoff amont connu (schéma issue #19) ; adoucisseur candidat `GLM53_MIXED_PREFILL_CHUNK=256` | `vllm-exl3/results/glm53-benchmark-20260829-115750.json` |
| 2026-08-29 | `…` (skip, re-mesure après ajout de `wall_seconds` au bench) | 9/9 ok, agrégé **50.0 tok/s** (wall exact 37.1 s), TTFT médian 2.60 s, **p99 14.79 s** | escalier reproduit (TTFT 0.83 → 2.6 → 8.6 → 9.2 → 14.8 : prefills en file derrière les décodes) ; remplace la mesure de 11h57 dont le wall manquait — le fallback `max(total_seconds)` du comparateur surestimait l'agrégé (72.1 tok/s estimés) et avait faussé une première comparaison | `vllm-exl3/results/glm53-benchmark-c4-chunkskip-20260829-173451.json` |
| 2026-08-29 | `c4-chunk-ab.sh --no-restart` sous `GLM53_MIXED_PREFILL_CHUNK=256` (même protocole C4) | 9/9 ok, agrégé **64.3 tok/s** (wall 30.7 s), TTFT médian 0.85 s, **p99 1.22 s** — verdict **PASS** : p99 ratio 0.083 (≤ 0.75), agrégé ratio 1.288 (≥ 0.95) | p99 TTFT **÷12** (14.8 → 1.2 s) et agrégé **+29 %** vs skip : le plafond mixte 256 élimine l'escalier sans rendre le goodput ; politique scheduler par défaut de la lane → **256** | `vllm-exl3/results/glm53-benchmark-c4-chunk256-20260829-173038.json` |
| 2026-08-29 | A/B qualité : `bench-glm53.py --prompts tests/ab_quality_prompts.jsonl --runs 1 --thinking on --save-content` (budgets ×6 identiques, temp 0) vs OpenRouter `z-ai/glm-5.3-flash` | **PASS RATE 4/4 = 4/4** (grader exécutable : 2 tâches code contre tests unitaires, 2 tâches JSON structurelles) ; `release_config` **bit-identique** (sha256 égal des deux côtés) | confirmation bout en bout de l'argument KLD (0.0246, EXL3-QUALITY.md) : le 4bpw passe les mêmes tests que la référence. Caveats : le côté référence est OpenRouter **provider Modal** (pas z.ai direct) et son reasoning est obligatoire (400 « Reasoning is mandatory… ») → thinking forcé des deux côtés ; 1 run/tâche. Ne pas citer les tok/s decode de ce run : en thinking le TTFT inclut le raisonnement, métrique non comparable aux rows DFlash2 | `vllm-exl3/results/ab-quality-20260829-233715.json` |
| 2026-08-29 | `bench_long_context.py --target-tokens 200000 --cold` | **ok, 3/3 aiguilles (sha256 exact), API saine** — 200 005 tokens, TTFT 229.5 s, prefill 871.3 tok/s e2e, décode 150.2 tok/s (réponse 40 tokens, petit échantillon) | pool 1.75M → 200k ≈ 11 %. Pas d'endpoint de reset sur ce build → le filler SESSION garantit le froid | `vllm-exl3/results/glm53-long-context-long-context-20260829-122213.json` |
| 2026-08-29 | `… --target-tokens 500000 --cold --label 500k-cold` | **ok, 3/3 aiguilles, API saine** — 500 011 tokens, TTFT 598.0 s, prefill 836.1 tok/s | pages des runs précédents encore résidentes ; −4 % de prefill vs 200k | `vllm-exl3/results/glm53-long-context-500k-cold-20260829-123606.json` |
| 2026-08-29 | `… --target-tokens 900000 --cold --label 900k-cold` | **ok, 3/3 aiguilles, API saine — 900 007 tokens froids** | TTFT 1138.4 s, prefill 790.6 tok/s ; 1.6M de pages cumulées résidentes dans le pool 1.75M | `vllm-exl3/results/glm53-long-context-900k-cold-20260829-125558.json` |
| 2026-08-29 | `… --target-tokens 990000 --cold --label 990k-cold` (après redémarrage) | **ok, 3/3 aiguilles, API saine — 990 007 tokens : la fenêtre 1M complète validée à froid** | TTFT 1231.9 s, prefill 803.6 tok/s (boot neuf, cache vide) ; `decode=null` = le garde de fenêtre minimale a correctement rejeté l'échantillon 40 tokens | `vllm-exl3/results/glm53-long-context-990k-cold-20260829-133322.json` |

Prefill le long de la rampe : 871 → 836 → 791 → 804 tok/s — quasi plat (pire
cas −9 %), la signature sparse-MLA.

### Checklist de promotion EXL3

Reprise du protocole DFlash2 de la lane sœur, plus les items propres à la
lane. **Le flip du défaut repo (ordre du README, merge `main`) n'arrive que
quand chaque case est cochée** ; d'ici là cette lane est le choix documenté
pour le long contexte et la fidélité de poids, et SGLang reste le défaut pour
les agents en rafales.

- zéro crash/retract pendant le balayage ; structured et prose relevés ;
  acceptation par position saine (pas d'effondrement tardif) ✅ 2026-08-29
- long contexte froid 3/3 aiguilles avec API saine après ✅ 2026-08-29
  (200k / 500k / 900k / 990k)
- réutilisation prefix-cache au voisinage du modèle de pages (eff ≥ 0.9)
  ✅ 2026-08-29
- comparaison C4 `GLM53_MIXED_PREFILL_CHUNK=256` : p99 TTFT nettement sous le
  p99 `skip` (16.7 s) sans rendre l'agrégé — décide de la politique scheduler
  par défaut de la lane ✅ 2026-08-29
  → verdict PASS (`tests/compare_c4.py`) : p99 14.79 → 1.22 s (ratio 0.083),
  agrégé 50.0 → 64.3 tok/s (ratio 1.288). Défaut de la lane → **256**
  (à poser dans `vllm-exl3/.env`, puis re-enregistrer les rows prefix-cache et
  long-context sous la nouvelle politique). Rows journal : 29/08 ci-dessus.
  → `vllm-exl3/scripts/c4-chunk-ab.sh` boote la politique candidate, rejoue le
  protocole C4 identique et applique le verdict via `vllm-exl3/tests/compare_c4.py`
  (p99 ≤ 0.75×, agrégé ≥ 0.95×). Ajouter la row ci-dessus après le run.
- soak tool-calling sous charge concurrente : aucun argument requis vide
  (issue amont #10 ouverte — validation + retry côté client d'ici là) ⬜
  → `vllm-exl3/tests/soak_tool_calls.py --agents 8 --turns 16` avec prefills
  froids concurrents (`--filler-words 8000`, `--salt` neuf à chaque run) ;
  exit 0 avec uniquement des événements récupérés = pass, plus les comptes
  bruts d'événements.
- soak OpenCode multi-jours sur trafic agent réel, redémarrage inclus ⬜
  → protocole + journal dans [EXL3-SOAK.md](EXL3-SOAK.md) (sonde quotidienne
  `vllm-exl3/scripts/soak-day.sh`).
- A/B qualité contre l'API officielle sur tâches de code/agent identiques
  (l'argument KLD mérite une confirmation bout en bout) ✅ 2026-08-29
  → PASS RATE **4/4 (EXL3 4bpw) = 4/4** (référence : OpenRouter
  `z-ai/glm-5.3-flash`, provider Modal — son reasoning étant obligatoire,
  thinking forcé des deux côtés, budgets ×6 identiques) ; `release_config`
  bit-identique. Caveat assumé : référence via OpenRouter tant qu'aucune clé
  z.ai directe — redo possible plus tard, row 29/08 ci-dessus.

Encore en attente sur ce kit : lecture `coldhit` sur un run prefix salé et les
deux soaks (la comparaison C4 et l'A/B qualité sont passés le 29/08, voir rows
et checklist ci-dessus).
