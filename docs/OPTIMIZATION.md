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
| `sm_121` (GB10) | pas de KV cache NVFP4 (exige `sm100f`) : FP8 e4m3 est le plancher et le plafond |
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
| `256k-graphs` | décode à 256k | la capture bs=1 est petite : réactiver les CUDA graphs à 262 144 tokens restaure le décode perdu en mode eager |
| `128k-ep1` | latence MoE | EP=1 (all-reduce pur) contre EP=2 (all-to-all sur RoCE) : même calcul, seul le motif de communication change |

Notes :

- le décodage spéculatif (MTP/NEXTN) est **lossless** : le modèle cible vérifie
  chaque token drafté. Ajuster `MTP_NUM_TOKENS` ne change pas la qualité ;
- `256k-graphs` : si la capture OOM, baissez `MEM_FRACTION_STATIC` à 0,88 ou
  retombez sur `256k` (eager) ;
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
| `SGLANG_ENABLE_SPEC_V2` | `0` | `1` = overlap scheduler sous spéculation ; cible directement le problème d'admission des prefills sous MTP | expérimental : exige `topk=1` (notre cas) ; mesurer avant adoption |

Exemple — essai combiné reproductible :

```bash
cp profiles/128k-batch4-mtp.env profiles/essai-mtp-specv2.env
printf 'SGLANG_ENABLE_SPEC_V2=1\n' >> profiles/essai-mtp-specv2.env
./stop-glm53.sh --profile 128k-batch4-mtp
./start-glm53.sh essai-mtp-specv2
./bench-glm53.py --runs 3 --concurrency 4
```

Surveillance utile pendant les mesures :

- `avg_spec_accept_length` dans les logs SGLang : > 4,5 → repassez à 5 étapes
  MTP ; < 3 → les étapes hautes sont du gaspillage ;
- « retract » ou « KV cache pool is full » → montez
  `SCHEDULE_CONSERVATIVENESS` (1.3 est un bon premier pas).

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
- **Requantification du `lm_head`** : sur GB10, un `lm_head` resté en BF16 est
  relu à chaque step de décode ; sa quantification NVFP4 isolée a montré
  jusqu'à +47 % de décode sur un modèle comparable. Vérifiez d'abord si le
  checkpoint l'exclut déjà de la quantification (`quantization_config.ignore`,
  cf. [AUDIT.md](AUDIT.md)). Une requantification complète OOMerait la mémoire
  unifiée ; seul le tenseur (~21 Go au pic) se re-quantifie. Exige une
  comparaison qualité A/B contre l'API officielle avant adoption.
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
| EP=1 vs EP=2 | lossless (même calcul, autre communication) |
| NCCL / RoCE / MTU | lossless |
| `SGLANG_ENABLE_SPEC_V2` | lossless en distribution ; à valider numériquement |
| KV FP8 e4m3 | déjà le socle validé de la recette |
| Requantification `lm_head` | risque faible, A/B obligatoire |
| RoPE/YaRN au-delà de la limite native | risque réel : exclu |

## Ordre de bataille recommandé

1. `256k-graphs` — plus gros gain dormant, zéro risque qualité ;
2. balayage `MTP_NUM_TOKENS` ∈ {2, 3, 4} à c=1 puis c=4 (profils
   `128k-batch4-mtp3`, `128k-batch2-mtp`) — cible le p99 de 45 s ;
3. `128k-batch4-8k` — TTFT/drain des rafales de sous-agents sans MTP ;
4. audit fabric (`ib_write_bw`, `NCCL_DEBUG=INFO`, MTU) puis `128k-ep1` ;
5. knobs niveau 2 un par un, en isolant chaque variable ;
6. seulement ensuite, niveau 4.

Ne combinez jamais deux changements dans une même mesure : sans isolement,
une régression est indetectable et un gain est inattribuable.
