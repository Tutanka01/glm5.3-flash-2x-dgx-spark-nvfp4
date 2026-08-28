# Optimisation : vitesse, concurrence et contexte

Ce document recense les leviers d'optimisation compatibles avec la recette
(2× GB10, TP=2, image SGLang épinglée), du plus simple au plus invasif. Il
complete [BENCHMARKS.md](BENCHMARKS.md) : chaque levier doit y gagner sa ligne
après mesure, et aucun ne modifie la distribution de sortie du modèle sauf
mention explicite.

## Contraintes physiques à connaître

| Contrainte | Conséquence |
|---|---|
| ~273 Go/s de bande passante LPDDR5X partagés CPU/GPU | le décode est borné par la mémoire : la quantification (octets/token) et la spéculation (tokens/lecture) sont les deux seuls leviers structurels |
| 119,6 Go de mémoire unifiée, poids ~90,6 Go/nœud en TP=2 | TP=2 est obligatoire ; pas de réplique DP par nœud (181,3 GiB > 128 Go) |
| `sm_121` (GB10) | pas de KV cache NVFP4 : FP8 e4m3 est le format compact disponible ; BF16 reste le témoin qualité |
| Lien RoCE point-à-point entre les nœuds | chaque all-reduce/all-to-all paie une latence ~40 µs, multipliée par couches × steps |

## Méthodologie

Chaque expérimentation suit le même cycle :

```bash
./stop-glm53.sh --profile <profil-actif>
./start-glm53.sh <profil-candidat>
./bench-glm53.py --runs 3 --concurrency N
./smoke-glm53.sh --profile <profil-candidat> --tools
```

1. comparez TTFT médian/p99, décode médian et goodput agrégé avec la ligne de
   référence du profil d'origine dans [BENCHMARKS.md](BENCHMARKS.md) ;
2. le smoke test vérifie la cohérence des réponses et du tool calling ;
3. notez la mesure dans [BENCHMARKS.md](BENCHMARKS.md) avec l'artefact JSON de
   `results/`, y compris les régressions ;
4. en cas de comportement anormal, capturez les logs des deux rangs
   (`./logs-glm53.sh --profile <candidat> --node both --tail 500`) avant de
   revenir au profil précédent.

## Niveau 1 — profils prêts à l'emploi

Ces profils ne nécessitent aucune modification du compose ou du `.env` :

| Profil | Cible | Hypothèse à mesurer |
|---|---|---|
| `128k-batch4-mtp3` | TTFT batché sous MTP | 3 étapes de spéculation raccourcissent les frontières de batch et réduisent le p99 de 45 s sans effacer le gain de décode |
| `128k-batch2-mtp` | compromis interactif | 2 requêtes concurrentes avec MTP 5 étapes : point d'équilibre entre 29 tok/s mono et le collapse TTFT à c=4 |
| `128k-batch4-8k` | drain des prefills longs | `MAX_NUM_BATCHED_TOKENS=8192` divise par deux les rounds de préfill ; surveiller le garde mémoire |
| `128k-mtp-ep1` | décode mono, motif de communication | combine MTP5/graphs avec EP=1 pour mesurer all-reduce TP contre all-to-all EP=2 |
| `128k-mtp-compile` | décode mono compilé | combine MTP5/graphs avec torch.compile borné à bs=1 ; démarrage plus long |
| `256k` | capacité froide à 240k | validé à 240 008 tokens : eager, sans MTP, chunk 1024 et statique 0,88 |
| `128k-dflash2` | décode spéculatif C1 | Mamba BF16/5 slots, draft 1B FA4 fenêtre 2048, graphes bs=1 |
| `128k-dflash2-c4` | décode spéculatif C4 | Mamba BF16/20 slots, statique 0,90, graphes bs=4 |
| `128k-dflash2-c8` | débit DFlash2 C5/C6 | Mamba BF16/40 slots, statique 0,90 (dérivé de 0,92 après guard trip au boot le 2026-08-28), graphes bs=8 |
| `128k-dflash2-flashinfer` | repli DFlash2 C1 | même gate mémoire avec attention draft FlashInfer |
| `256k-dflash2-eager` | DFlash2 + froid 240K | pression maximale, graphes coupés, chunk 2048, Mamba BF16/5, statique 0,88 |
| `256k-graphs` | décode court avec limite 256k | ✅ 14,4 tok/s sur petits prompts ; capture bs=1 validée, capacité froide 256k non testée |
| `256k-mtp` | décode court uniquement | quarantaine après `SIGKILL` du scheduler pendant le test froid 240k ; le client refuse >128k |
| `384k-quality` | reproduction >256k sans compression KV supplémentaire | quarantaine : KV BF16, MTP, graphs et CP=2, aucune capacité prouvée |
| `512k-mtp-eager` | reproduction >256k | quarantaine : retire un défaut de replay connu, pas le risque mémoire |
| `512k-mtp-cp` | reproduction >256k | quarantaine : MTP + graphs + CP=2, non validé sur ce runtime |
| `128k-ep1` | latence MoE | EP=1 (all-reduce pur) contre EP=2 (all-to-all sur RoCE) : même calcul, seul le motif de communication change |

Notes :

- le décodage spéculatif (MTP/NEXTN) est **lossless par construction** : le
  modèle cible vérifie chaque token drafté. Ajuster `MTP_NUM_TOKENS` ne crée pas
  une nouvelle quantification ; le smoke et les tests déterministes restent
  obligatoires pour détecter un bug d'implémentation ;
- pour toute capacité froide à 240k, utilisez `256k`. Ne transposez pas MTP ou
  les graphes avant que ce run n'ait produit un JSON `ok=true` ;
- une limite configurée n'alloue pas `contexte × requêtes` de KV à l'avance :
  SGLang remplit un pool issu de la mémoire restante. Comparez la ligne
  `KV cache pool` des logs et testez une vraie requête longue ;
- `128k-ep1` : EP=2 reste la configuration auditée. Si `128k-ep1` ne démarre
  pas ou produit un smoke test incohérent, l'image épinglée ne couvre pas ce
  chemin — n'insistez pas.

## Niveau 2 — knobs d'exécution optionnels

Ces variables se posent dans `.env.glm53` ou dans une copie de profil
(`cp profiles/128k-batch4.env profiles/essai.env` puis ajout en fin de
fichier). Les défauts reproduisent exactement le comportement validé :

| Variable | Défaut | Effet | Risque |
|---|---|---|---|
| `EP_SIZE` | `2` | `1` = TP pur (all-reduce), `2` = experts parallèles (all-to-all) | nul sur la qualité ; à benchmarker |
| `ENABLE_TORCH_COMPILE` | `0` | `1` = torch.compile sur le chemin de décode (`TORCH_COMPILE_MAX_BS` borne le batch compilé) | nul ; démarrage plus long |
| `ENABLE_MIXED_CHUNK` | `0` | `1` = mélange des chunks de prefill aux batches de décode (TTFT batché sans MTP) | incompatible avec MTP : le validateur refuse la combinaison |
| `SCHEDULE_CONSERVATIVENESS` | `1.0` | > 1.0 = admission plus conservatrice, évite les retracts si les logs montrent « KV cache pool is full » | nul |
| `ENABLE_PREFILL_CP` | `0` | `1` + `ATTN_CP_SIZE=2` partage le long préfill DSA entre les deux rangs | expérimental sur ce runtime ; test froid obligatoire |
| `CP_STRATEGY` | `interleave` | distribution round-robin des tokens, compatible multi-batch dans SGLang | garder `interleave` pour le premier audit |

La révision SGLang épinglée exécute déjà tous les workers EAGLE/NEXTN avec Spec
V2 et l'overlap scheduler. L'ancienne variable `SGLANG_ENABLE_SPEC_V2` y est
retirée : la définir n'active rien et produit seulement un avertissement. Le
TTFT batché mesuré à 45 s p99 inclut donc déjà Spec V2 ; les leviers utiles sont
la profondeur MTP, la concurrence et la politique d'admission.

Surveillance utile pendant les mesures :

- `avg_spec_accept_length` dans les logs SGLang : > 4,5 → repassez à 5 étapes
  MTP ; < 3 → les étapes hautes sont du gaspillage ;
- « retract » ou « KV cache pool is full » → montez
  `SCHEDULE_CONSERVATIVENESS` (1.3 est un bon premier pas).

## Palier long-contexte : capacité réelle, pas limite déclarée

Le checkpoint annonce 1 048 576 tokens nativement : aucun RoPE scaling n'est
nécessaire jusqu'à cette limite. En revanche, deux limites d'exécution sont
distinctes :

1. le nombre de tokens du pool KV réellement alloué après chargement/capture ;
2. le chemin de CUDA graph replay. [SGLang #36550](https://github.com/sgl-project/sglang/issues/36550)
   reproduit un abort au premier token après un préfill froid >262 144, alors
   que le même prompt passe en eager. Le ticket montre aussi que le préfill CP=2
   masque le défaut au moins jusqu'à 428k en divisant la longueur vue par rang.

Le profil rapide ne doit donc pas être validé avec `bench-glm53.py`, dont les
prompts sont courts, mais avec :

```bash
./start-glm53.sh 512k-mtp-cp
./bench-long-context.py --allow-unsafe-profile --target-tokens 300000 --cold --label 512k-mtp-cp-300k
./bench-long-context.py --allow-unsafe-profile --target-tokens 400000 --cold --label 512k-mtp-cp-400k
./bench-long-context.py --allow-unsafe-profile --target-tokens 480000 --cold --label 512k-mtp-cp-480k
```

Le client place trois aiguilles, enregistre le nombre de tokens réellement vu
après le template, et vérifie la santé du serveur après la réponse. Si un palier
échoue ou tue un rang, conservez les logs et passez à `512k-mtp-eager`. Le profil
eager garde MTP : il sacrifie le gain de capture, pas la vérification des tokens.

## Niveau 3 — fabric RoCE

Le TTFT batché et la latence de décode intègrent la communication TP à chaque
couche. Trois vérifications côté hôte, **d'abord hors recette**, puis en bench
cluster :

1. **Débit brut du lien** — certains GB10 sortent d'usine bridés :

   ```bash
   ib_write_bw -d <HCA> --report_gbits     # attendu ~100 Gb/s par rail
   ```

2. **Fusion des rails** — si votre câble QSFP expose deux ports par nœud
   (`ibdev2netdev` en liste deux) :

   ```bash
   NCCL_IB_MERGE_NICS=1
   NCCL_IB_HCA=rocep1s0f1,rocep1s0f2   # les deux rails
   ```

   Mesure de référence indépendante : ~13,5 → ~22 Go/s en all-reduce. Si votre
   fabric est mono-lien, ce knob ne change rien.

3. **GPUDirect RDMA et trames jumbo** :

   ```bash
   NCCL_NET_GDR_LEVEL=SYS
   NCCL_DMABUF_ENABLE=1
   NCCL_IB_QPS_PER_CONNECTION=4
   ip link set <iface-fabric> mtu 9000          # sur les deux nœuds
   ```

   Vérifiez `NCCL_DEBUG=INFO` : la ligne NCCL doit mentionner GDR actif après
   changement. Un MTU 9000 doit être valide de bout en bout (les deux extrémités).

Ces quatre variables sont déjà passées au conteneur par le compose : une valeur
vide ou absente laisse NCCL à son comportement par défaut validé
(`NCCL_CROSS_NIC=1`, `NVLS=0`, `CUMEM=0`, 4 canaux — voir
[NETWORK.md](NETWORK.md) pour le socle à ne pas modifier).

## Niveau 4 — au-delà de la recette (risque maîtrisé mais réel)

- **Ré-audit d'une image SGLang plus récente** : le compose neutralise
  `SGLANG_OPT_USE_TOPK_V2`, `SGLANG_OPT_FP8_WO_A_GEMM` et
  `SGLANG_OPT_DEEPGEMM_HC_PRENORM` (cassés sur `sm_121` dans l'image épinglée).
  Une image suivante peut les réactiver : ne le faites que via une nouvelle
  ligne d'audit dans [AUDIT.md](AUDIT.md), pins digest à l'appui.
- **Requantification du `lm_head`** : levier de vitesse possible, mais il ajoute
  une approximation précisément sur la projection de sortie. Il est exclu de
  cet ordre de bataille puisque l'objectif impose zéro nouveau compromis
  qualité.
- **Extension de contexte au-delà de la limite native** (RoPE/YaRN) : dégrade
  la qualité de façon mesurable. La limite native du checkpoint est le plafond
  assumé de cette recette ; `MAX_MODEL_LEN` ne doit pas la dépasser.

## Ce qui ne s'applique pas au GB10

| Piste | Pourquoi ce n'est pas applicable |
|---|---|
| HiCache / `--enable-hierarchical-cache` | la mémoire « host » est le même pool unifié : aucun gain de capacité, et concurrence avec le garde OOM de 6 Go |
| KV cache NVFP4 | exige `sm100f` ; GB10 = `sm_121` |
| Répliques DP (1 par nœud) | 181,3 GiB de poids > 128 Go unifiés par nœud |
| Déconsolidation préfill/décode (PD) | chaque instance devrait répliquer les 181,3 GiB |
| NVLS / Switch NCCL | inutile à deux nœuds sans NVSwitch |

## Risque qualité par levier

| Levier | Qualité |
|---|---|
| MTP (toutes valeurs de `MTP_NUM_TOKENS`) | lossless (vérification par le modèle cible) |
| CUDA graphs, torch.compile, chunked prefill, conservativeness | lossless |
| Préfill context parallel | même modèle/calcul ; ordre de réduction différent, test numérique obligatoire |
| EP=1 vs EP=2 | lossless (même calcul, autre communication) |
| NCCL / RoCE / MTU | lossless |
| KV FP8 e4m3 | compression numérique déjà présente dans le socle rapide ; pas strictement équivalente au BF16 |
| KV BF16 (`384k-quality`) | aucune quantification KV supplémentaire ; capacité environ réduite |
| Requantification `lm_head` | perte potentielle : exclue |
| RoPE/YaRN au-delà de la limite native | risque réel : exclu |

## Ordre de bataille recommandé

1. `256k` — test froid à 240k en configuration de fiabilité ; conserver le JSON
   et vérifier que l'API reste saine (`./start-glm53.sh 256k`) ;
2. `256k-graphs` puis `256k-mtp` — benchmarks courts uniquement. Aucun long
   prompt tant qu'une recette mémoire distincte n'a pas été validée ;
3. `512k-mtp-eager` — reproduction expérimentale avec
   `--allow-unsafe-profile`, jamais une étape de production automatique ;
4. `512k-mtp-cp` — mêmes paliers avec graphs + CP=2, puis comparaison directe
   du TTFT et du décode au profil eager ;
5. `384k-quality` — test froid 360k en KV BF16 et comparaison de qualité/vitesse
   au socle FP8 ;
6. `128k-mtp-ep1`, puis `128k-mtp-compile` — deux ablations mono-flux contre
   la référence MTP5 à ~29 tok/s ; ne les combinez que si chacune gagne seule ;
7. balayage `MTP_NUM_TOKENS` ∈ {2, 3, 4} à c=1 puis c=4 (profils
   `128k-batch4-mtp3`, `128k-batch2-mtp`) — cible le p99 de 45 s ;
8. `128k-batch4-8k` — TTFT/drain des rafales de sous-agents sans MTP ;
9. audit fabric (`ib_write_bw`, `NCCL_DEBUG=INFO`, MTU) ;
10. knobs niveau 2 un par un, en isolant chaque variable ;
11. seulement ensuite, ré-audit d'un runtime plus récent.

La piste DFlash2 dispose maintenant de trois profils expérimentaux. Préparez
l'image et le draft avec `./prepare-dflash2.sh`, puis suivez le sas décrit dans
[DFLASH2.md](DFLASH2.md). Les patches KV du port vLLM ne sont pas copiés.

Ne combinez jamais deux changements dans une même mesure : sans isolement,
une régression est indetectable et un gain est inattribuable.
