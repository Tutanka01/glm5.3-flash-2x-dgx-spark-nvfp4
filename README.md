# GLM-5.3-Flash NVFP4 sur 2× DGX Spark / PGX

Cette recette déploie `LibertAIDAI/GLM-5.3-Flash-NVFP4` avec SGLang sur
**deux** machines GB10 en tensor parallel (`TP=2`). Une seule commande lancée
sur le head synchronise le worker, contrôle les deux machines, charge le modèle
et expose une API compatible OpenAI.

> **État actuel :** le runtime épinglé produit des réponses cohérentes et les
> profils 32K/128K ont été mesurés sur deux GB10. Le long contexte est validé
> à froid : 240 008 tokens sans spéculation (`256k`) et 200 012 tokens avec
> DFlash2 (`256k-dflash2-eager`, décode ×3). Les profils 384K/512K restent des
> candidats expérimentaux tant qu'ils n'ont pas passé le benchmark froid
> long-contexte sur votre cluster. Le checkpoint NVFP4 est communautaire ;
> ne confondez pas validation technique et équivalence qualité avec l'API
> officielle.

## À lire avant de commencer

- Cette recette exige **2× DGX Spark/ThinkStation PGX GB10** sous Linux ARM64,
  avec un driver compatible CUDA 13, Docker et Docker Compose v2. Elle ne lance
  pas ce checkpoint 320B sur une seule machine.
- Toutes les commandes de ce guide s'exécutent sur le **head**. Il n'est pas
  nécessaire de cloner manuellement le dépôt sur le worker : les scripts y
  copient les fichiers nécessaires par SSH.
- Le head doit disposer de `git`, `python3`, `curl`, `ssh` et `scp`. Les deux
  machines doivent permettre à l'utilisateur courant d'accéder à Docker et au
  GPU avec `nvidia-smi`.
- Chaque machine stocke une copie complète du checkpoint, soit environ
  **181,3 GiB**. Prévoyez au moins **205 GiB libres par nœud**.
- Le premier téléchargement peut être long. Un démarrage déjà préparé prend
  ensuite souvent 10 à 15 minutes sur le cluster de référence ; le script
  attend jusqu'à une heure. Ne l'interrompez pas tant qu'il affiche une
  progression de chargement.
- Le chemin RoCE entre les deux machines doit déjà fonctionner. Les scripts
  peuvent le vérifier, mais ils ne peuvent pas inventer les bonnes interfaces
  et adresses pour votre câblage.
- Arrêtez les autres modèles gourmands en mémoire sur les deux nœuds. Le swap
  désactivé est recommandé pendant la première mise en service.

## Parcours guidé : de zéro à une API validée

Suivez les étapes dans l'ordre. Ne passez au profil suivant que lorsque le
résultat attendu de l'étape courante est obtenu.

### 1. Préparer le head et l'accès au worker

Sur le head :

```bash
git clone https://github.com/Tutanka01/glm5.3-flash-2x-dgx-spark-nvfp4.git
cd glm5.3-flash-2x-dgx-spark-nvfp4
docker compose version
python3 --version
curl --version
```

Vérifiez ensuite que le head peut joindre le worker sans demander de mot de
passe. Remplacez la destination par celle que vous utiliserez dans
`WORKER_HOST` :

```bash
ssh -o BatchMode=yes utilisateur@worker true
```

Si cette dernière commande échoue, configurez d'abord une clé SSH, par exemple :

```bash
ssh-keygen -t ed25519
ssh-copy-id utilisateur@worker
ssh -o BatchMode=yes utilisateur@worker true
```

Le lancement automatique ne fonctionnera pas avec une invite de mot de passe
interactive.

### 2. Créer la configuration du cluster

```bash
cp .env.glm53.example .env.glm53
chmod 600 .env.glm53
$EDITOR .env.glm53
```

Les valeurs à adapter sont regroupées au début du fichier :

| Groupe | Variables principales | Signification |
|---|---|---|
| Accès worker | `WORKER_HOST`, `WORKER_DIR` | destination SSH et dossier absolu créé sur le worker |
| Fabric | `HEAD_FABRIC_IP`, `WORKER_FABRIC_IP`, `MASTER_ADDR` | adresses privées utilisées entre les deux rangs |
| Interfaces | `NCCL_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME` | interface réseau portant l'IP fabric sur le head |
| RDMA | `NCCL_IB_HCA`, `NCCL_IB_ADDR_RANGE` | HCA RoCE et sous-réseau du fabric |
| Worker différent | variables préfixées `WORKER_` | uniquement si les noms ou chemins diffèrent sur le worker |
| API cliente | `API_ADVERTISE_HOST` | IP de management utilisée par OpenCode et les autres clients |

Pour identifier les interfaces et le HCA sur chaque machine :

```bash
ip -br -4 addr
ibdev2netdev
```

Reprenez de préférence les valeurs d'une recette TP=2 déjà fonctionnelle sur
ces deux machines. Le détail et les exemples se trouvent dans
[docs/NETWORK.md](docs/NETWORK.md).

Conservez sans modification `MODEL_ID`, `MODEL_REVISION` et
`GLM53_RUNTIME_IMAGE`. N'ajoutez pas `NCCL_IB_GID_INDEX` : ce runtime sélectionne
automatiquement un GID RoCE v2.

### 3. Valider la configuration et les deux machines

```bash
./validate-glm53.sh --config-only
./doctor-glm53.sh
```

La première commande refuse les valeurs manquantes, les anciens pins et les
combinaisons incohérentes. Le doctor vérifie ensuite, sur les deux nœuds,
l'architecture ARM64, le GPU, Docker, la mémoire, le disque, les interfaces,
les routes, le HCA et les GID RoCE.

Résultat attendu :

```text
[glm53] Configuration is valid
[glm53] Two-node doctor completed successfully
```

Un `[FAIL]` est bloquant. Un `[WARN]` doit être compris avant de continuer ; les
cas connus sont expliqués dans [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

### 4. Télécharger et vérifier le runtime et le checkpoint

```bash
./prepare-glm53.sh
```

Cette commande vérifie les pins distants, synchronise la recette, tire l'image
sur les deux machines, télécharge le snapshot complet sur chacune et valide les
120 shards ainsi que les métadonnées NVFP4. Par défaut, les téléchargements sont
séquentiels pour ne pas saturer une connexion Internet partagée. Si chaque
machine possède réellement son propre accès rapide :

```bash
PREPARE_PARALLEL=1 ./prepare-glm53.sh
```

Résultat attendu :

```text
[glm53] Preparation complete: pinned image and checkpoint validated on both nodes
```

Les démarrages suivants réutilisent les fichiers locaux et restent hors ligne.

### 5. Faire le premier démarrage sûr

Commencez toujours par `32k`, sans MTP et avec une seule requête :

```bash
./start-glm53.sh 32k
```

Le script lance le worker, puis le head, attend `/v1/models` et termine par une
génération déterministe. Des messages de compilation ou de capture CUDA sont
normaux. Le succès est uniquement confirmé par les dernières lignes :

```text
[glm53] Basic chat smoke test passed
GLM-5.3-Flash is serving at http://<ADRESSE_HEAD>:8888/v1
```

Si un rang tombe avant ce point, le launcher collecte les journaux et arrête les
deux conteneurs. Ne lancez pas un benchmark pendant le chargement.

### 6. Vérifier le service avant de l'utiliser

```bash
./status-glm53.sh 32k
./smoke-glm53.sh --profile 32k --tools
./bench-glm53.py --runs 3
```

Le statut doit montrer les deux conteneurs actifs, le bon identifiant de modèle
et les deux lignes suivantes :

```text
ready model=glm-5.3-flash-nvfp4 max_model_len=32768
tokenizer ready
```

Selon la version de SGLang, `max_model_len` peut être affiché comme `unknown` ;
le bon identifiant de modèle et `tokenizer ready` restent obligatoires.

Le smoke test vérifie le chat et le tool calling. Le benchmark doit terminer
avec toutes les requêtes réussies. Si vous avez défini `API_KEY` dans
`.env.glm53`, exportez la même variable dans le terminal avant d'utiliser les
clients Python de benchmark :

```bash
export API_KEY='même valeur que dans .env.glm53'
```

À ce stade seulement, le déploiement de base est validé.

## Choisir et valider un profil de production

Le meilleur profil dépend du type de charge. Les recommandations issues des
mesures actuelles sont :

| Besoin | Profil de départ | Pourquoi |
|---|---|---|
| validation initiale ou faible trafic | `32k` | chemin le plus simple, une requête |
| sous-agents et requêtes simultanées | `128k-dflash2-c8` | 86,0 tok/s agrégés à C6, TTFT p99 0,82 s mesurés ; `128k-dflash2-c4` (67,2 à C4) et `128k-batch4` (31,5) en replis éprouvés |
| un flux interactif prioritaire | `128k-dflash2` | 37,2 tok/s mesurés, TTFT intact ; `128k-batch4-mtp` (29 tok/s) en repli épinglé |
| diagnostic CUDA Graphs | `32k-eager` | retire uniquement les graphes |
| contexte long au quotidien (défaut) | `256k-dflash2-eager` | 200 012 tokens froids validés, décode 39,6 tok/s ; exige `./prepare-dflash2.sh` une fois ; drafter sous licence CC BY-NC-ND (recherche/évaluation) |
| contexte maximal (> 200K) | `256k` | validé à 240 008 tokens froids : eager, sans MTP, préfill 1024, statique 0,88 ; décode 13,5 tok/s |
| contexte supérieur à 256K | aucun | profils 384/512K en quarantaine, non fiables sur TP2/GB10 |
| décode mono-flux de prochaine génération | `128k-dflash2` | mesuré 37,2 tok/s mono et 86,0 tok/s agrégés à C6 (profil c8) ; `experimental` jusqu'à la validation acceptation + stabilité |

MTP accélère fortement un seul décode, mais dégrade le TTFT lors de rafales
concurrentes. Pour plusieurs sous-agents, préférez donc `128k-batch4` sans MTP.
La mesure du 28 août montre que DFlash2 combine les deux régimes : le profil
`128k-dflash2-c8` atteint 86,0 tok/s agrégés à C6 avec un TTFT p99 de 0,82 s,
mais s'appuie sur l'image expérimentale DFlash2 et son drafter sous licence
CC BY-NC-ND.

Pour changer de profil, arrêtez toujours les deux rangs avant le redémarrage :

```bash
./stop-glm53.sh --profile 32k
./start-glm53.sh 128k-batch4
./status-glm53.sh 128k-batch4
./smoke-glm53.sh --profile 128k-batch4 --tools
./bench-glm53.py --runs 3 --concurrency 4
```

Avant d'annoncer une capacité de contexte, testez-la avec un vrai prompt froid.
Le défaut long contexte est `256k-dflash2-eager` (200 012 tokens froids validés,
décode 39,6 tok/s) ; la préparation DFlash2 se fait une fois :

```bash
./prepare-dflash2.sh 256k-dflash2-eager
./start-glm53.sh 256k-dflash2-eager
./status-glm53.sh 256k-dflash2-eager
./bench-long-context.py --allow-unsafe-profile --target-tokens 200000 --cold \
  --label 256k-dflash2-200000
./status-glm53.sh 256k-dflash2-eager
```

Au-delà de 200K et jusqu'à 240K, utilisez exclusivement le profil de fiabilité
`256k` (240 008 tokens froids validés, sans spéculation) :

```bash
./stop-glm53.sh --profile 256k-dflash2-eager  # si c'est le profil actuellement lancé
./start-glm53.sh 256k
./status-glm53.sh 256k
./bench-long-context.py --target-tokens 240000 --cold --label 256k-safe
./status-glm53.sh 256k
```

La capacité n'est validée que si `ok=True`, les trois aiguilles sont retrouvées
et l'API reste saine après la requête. Un petit benchmark lancé avec une limite
serveur à 256K ne prouve pas que 240K tokens réels fonctionnent.

`--label` nomme uniquement le fichier de résultat : il ne sélectionne jamais le
profil serveur. Au-dessus de 128K, le client inspecte donc le conteneur réellement
lancé et exige par défaut les graphes CUDA désactivés, MTP coupé, un chunk
inférieur ou égal à 2048, une seule requête et
`MEM_FRACTION_STATIC<=0.88`. Le contournement
`--allow-unsafe-profile` est réservé aux reproductions de crash assumées.

Les profils portent également un niveau de preuve affiché au démarrage. Rien
n'est bloqué : `capacity-candidate`, `experimental` et `quarantined` démarrent
directement, avec un avertissement clair. Le client de long contexte conserve
`--allow-unsafe-profile` afin qu'un test destructif reste explicite dans le
fichier JSON et dans l'historique shell.

Si un rang tombe, ne redémarrez pas tout de suite :

```bash
./collect-glm53-report.sh --profile 256k-mtp
```

Cette commande fabrique une archive dans `results/diagnostics/` avec les logs
des deux rangs, l'état Docker, la mémoire disponible et les traces OOM du
kernel. Elle exclut volontairement l'environnement Docker et `API_KEY`.

### Checklist avant exposition en production

- générez un secret long, par exemple avec `openssl rand -hex 32`, puis activez
  `API_KEY` dans `.env.glm53` et configurez la même valeur dans chaque client ;
- limitez le port `8888/tcp` au réseau de confiance avec le pare-feu du head ;
- conservez `.env.glm53` en permission `0600` et hors de Git ;
- gardez les pins du modèle et de l'image inchangés ; `prepare-glm53.sh` bloque
  volontairement une dérive non auditée ;
- conservez `RESTART_POLICY=no` : un seul rang ne doit pas redémarrer isolément.
  Après un reboot, attendez le réseau, Docker et SSH, puis relancez le profil
  choisi avec `start-glm53.sh` depuis le head ;
- surveillez les deux rangs, `/v1/models`, le tokenizer et une petite génération,
  pas seulement l'ouverture du port HTTP ;
- archivez le JSON du benchmark accepté et le nom exact du profil déployé.

`API_HOST=0.0.0.0` est nécessaire au fonctionnement de cette recette entre les
nœuds. La protection doit donc être assurée par la clé API et le filtrage réseau,
pas en remplaçant cette valeur par `127.0.0.1`.

## Exploitation courante

```bash
# État des deux rangs, de /v1/models et du tokenizer
./status-glm53.sh 128k-batch4

# Journaux récents du head et du worker
./logs-glm53.sh --profile 128k-batch4 --node both --tail 300

# Test chat + tool calling
./smoke-glm53.sh --profile 128k-batch4 --tools

# Arrêt coordonné des deux rangs
./stop-glm53.sh --profile 128k-batch4
```

Pour suivre les logs, sélectionnez un seul nœud. Interrompre cette commande de
suivi avec `Ctrl+C` ne doit pas être confondu avec l'arrêt du service :

```bash
./logs-glm53.sh --profile 128k-batch4 --node head --tail 300 --follow
```

Un journal contenant `SIGTERM received` indique qu'une commande externe a
demandé l'arrêt. Ce n'est pas, à lui seul, un crash CUDA.

## Architecture

```text
head (rank 0)                              worker (rank 1, headless)
start-glm53.sh ─────── SSH, worker-first ───────► SGLang
      │                                           │
      └────────── TP=2 / NCCL + Gloo / RoCE ──────┘
      │
      └── http://HEAD:8888/v1  (API sur le head uniquement)
```

Chaque nœud conserve le checkpoint complet sur disque. Au chargement, SGLang
répartit environ 181,3 GiB de tenseurs entre les deux rangs, soit une charge
idéale proche de 90,64 GiB de poids indexés par nœud. Le head synchronise la
configuration, valide les snapshots, démarre le worker en premier, lance son
propre rang, attend l'API et exécute le smoke test automatique.

### Migration depuis l'ancienne recette vLLM

Si `.env.glm53` contient encore `GLM53_VLLM_IMAGE` ou `NCCL_IB_GID_INDEX`, ne
le corrigez pas morceau par morceau. Repartez de l'exemple courant :

```bash
cp .env.glm53 .env.glm53.before-sglang
cp .env.glm53.example .env.glm53
chmod 600 .env.glm53
$EDITOR .env.glm53
```

Recopiez uniquement les paramètres SSH, les chemins de cache et les valeurs
réseau confirmées. Conservez les pins actuels du modèle et du runtime.

## Profils de lancement

| Profil | Contexte | Requêtes | MoE | Graphes | MTP | Usage / état |
|---|---:|---:|---|---|---:|---|
| `32k` | 32 768 | 1 | FlashInfer CUTLASS | oui | non | premier démarrage |
| `32k-eager` | 32 768 | 1 | FlashInfer CUTLASS | non | non | diagnostic sans CUDA graphs |
| `32k-mtp` | 32 768 | 1 | FlashInfer CUTLASS | oui | 5 étapes | MTP mesuré à environ 29 tok/s |
| `32k-batch4` | 32 768 | 4 | FlashInfer CUTLASS | oui | non | sous-agents OpenCode |
| `32k-batch8` | 32 768 | 8 | FlashInfer CUTLASS | oui | non | concurrence élevée, expérimental |
| `64k` | 65 536 | 1 | FlashInfer CUTLASS | oui | non | deuxième étape |
| `128k` | 131 072 | 1 | FlashInfer CUTLASS | oui | non | long contexte mono-requête |
| `128k-batch4` | 131 072 | 4 | FlashInfer CUTLASS | oui | non | mesuré, conseillé pour les sous-agents |
| `128k-batch4-8k` | 131 072 | 4 | FlashInfer CUTLASS | oui | non | prefill 8192 expérimental |
| `128k-batch4-mtp` | 131 072 | 4 | FlashInfer CUTLASS | oui | 5 étapes | mono-flux mesuré ×2, mauvais p99 en rafale |
| `128k-batch4-mtp3` | 131 072 | 4 | FlashInfer CUTLASS | oui | 3 étapes | compromis MTP batché à mesurer |
| `128k-batch2-mtp` | 131 072 | 2 | FlashInfer CUTLASS | oui | 5 étapes | compromis interactif à mesurer |
| `128k-batch8` | 131 072 | 8 | FlashInfer CUTLASS | oui | non | longs contextes + forte concurrence |
| `128k-ep1` | 131 072 | 4 | FlashInfer CUTLASS | oui | non | ablation TP/EP expérimentale |
| `128k-mtp-ep1` | 131 072 | 1 | FlashInfer CUTLASS | oui | 5 étapes | ablation MTP + EP=1 expérimentale |
| `128k-mtp-compile` | 131 072 | 1 | FlashInfer CUTLASS | oui | 5 étapes | ablation torch.compile expérimentale |
| `128k-dflash2` | 131 072 | 1 | FlashInfer CUTLASS | oui | DFlash2 | démarrage C1 : Mamba BF16/5 slots, draft FA4 fenêtre 2048 ; 37,2 tok/s mesurés |
| `128k-dflash2-c4` | 131 072 | 4 | FlashInfer CUTLASS | oui | DFlash2 | concurrence C4 : Mamba BF16/20 slots, statique 0,90 ; 67,2 tok/s agrégés mesurés |
| `128k-dflash2-c8` | 131 072 | 8 | FlashInfer CUTLASS | oui | DFlash2 | stress graphes bs=8 : Mamba BF16/40 slots, statique 0,90 ; 0,92 n'a jamais atteint la readiness (guard trip le 2026-08-28) ; 86,0 tok/s agrégés mesurés à C6 |
| `128k-dflash2-flashinfer` | 131 072 | 1 | FlashInfer CUTLASS | oui | DFlash2 | repli C1 si FA4 échoue sur SM121 |
| `256k` | 262 144 | 1 | FlashInfer CUTLASS | non | non | 240 008 tokens froids validés : chunk 1024, statique 0,88, récupération 3/3 |
| `256k-graphs` | 262 144 | 1 | FlashInfer CUTLASS | oui | non | quarantaine ; capture bs=1 seule validée |
| `256k-mtp` | 262 144 | 1 | FlashInfer CUTLASS | oui | 5 étapes | quarantaine : décode court seulement, froid >128K refusé |
| `256k-dflash2-eager` | 262 144 | 1 | FlashInfer CUTLASS | non | DFlash2 | défaut long contexte : draft 1B + froid long, Mamba BF16/5, statique 0,88 ; 200K froid validé le 2026-08-28 (décode ×3 vs `256k`), plafond pool ≈ 210K tokens |
| `384k-quality` | 393 216 | 1 | FlashInfer CUTLASS | oui | 5 étapes | quarantaine ; aucune capacité froide prouvée |
| `512k-mtp-eager` | 524 288 | 1 | FlashInfer CUTLASS | non | 5 étapes | quarantaine ; retirer les graphes ne suffit pas à prouver 512K |
| `512k-mtp-cp` | 524 288 | 1 | FlashInfer CUTLASS | oui | 5 étapes | quarantaine ; CP=2 non validé sur ce runtime |

Les noms de profils sont des abréviations : `128k` correspond à `MAX_MODEL_LEN=131072`, soit 131 072 tokens (128 × 1024).

Arrêtez toujours le profil actif avant d'en charger un autre :

```bash
./stop-glm53.sh --profile 32k
./start-glm53.sh 32k-batch4
```

Pour augmenter la fenêtre, progressez dans l'ordre `32k` → `64k` → `128k` →
`256k` et validez chaque palier. Pour tester MTP, comparez d'abord `32k` à
`32k-mtp`, puis transposez le réglage au contexte voulu. Utilisez `32k-eager`
uniquement pour isoler un problème de capture ou de replay CUDA graphs.

## Concurrence et sous-agents

Le profil `32k` n'accepte qu'une seule requête (`MAX_NUM_SEQS=1`) : c'est un choix de fiabilité pour le premier démarrage, pas une limite du modèle. Les profils `32k-batch4` et `32k-batch8` lèvent cette limite et permettent à OpenCode de lancer plusieurs sous-agents en parallèle.

À savoir en passant au batché :

- le débit de décode **par requête** baisse légèrement (la bande passante mémoire est partagée), mais le débit **agrégé** augmente nettement ;
- le radix cache de SGLang est actif : les requêtes qui partagent un préfixe identique (system prompt des sous-agents) réutilisent le KV et un préfill quasi gratuit ;
- si le garde mémoire se déclenche ou que le préfill échoue sur `32k-batch8`, revenez à `32k-batch4` ou remettez `MAX_NUM_BATCHED_TOKENS=4096`.

Les profils `128k-batch4` et `128k-batch8` combinent 131 072 tokens de contexte et la concurrence. Le pool KV est partagé entre les requêtes actives : si quatre conversations de 131k ne tiennent pas simultanément, SGLang met simplement les requêtes en excès en file d'attente au lieu de les rejeter. En usage agentique réel, peu de conversations remplissent tout le contexte, donc la concurrence utile est généralement supérieure au cas le pire.

Le MTP est désormais mesuré sur cluster (voir [BENCHMARKS.md](docs/BENCHMARKS.md)) : il double le débit de décode mono-flux (14,5 → 29,0 tok/s) mais dégrade fortement le TTFT en batché (p99 45 s à concurrence 4) car l'admission des nouveaux prefills attend les frontières de batch. En pratique : `128k-batch4-mtp` pour l'usage interactif mono-flux, `128k-batch4` sans MTP pour les rafales de sous-agents. Les profils `128k-batch4-mtp3` (3 étapes) et `128k-batch2-mtp` (concurrence 2) explorent le point d'équilibre entre ces deux régimes. La mesure DFlash2 du 28 août les rend toutefois moins urgents : le balayage `128k-dflash2-c8` monte à 86,0 tok/s agrégés à C6 avec un TTFT p99 de 0,82 s, devant les deux régimes MTP, sur l'image expérimentale (voir [BENCHMARKS.md](docs/BENCHMARKS.md)).

Le résultat `256k-graphs` à 14,4 tok/s utilise les petits prompts du benchmark standard : il valide le démarrage, la capture bs=1 et le décode court avec une limite configurée à 262 144, mais **pas** un préfill froid de 256k. Le run froid `256k-mtp` du 27 août s'est figé vers 164K tokens traités, puis le scheduler a reçu `SIGKILL` ; ce profil est donc mis en quarantaine. À l'inverse, `256k` a réussi le 28 août avec 240 008 tokens après template, TTFT 204,59 s, préfill 1 173,12 tok/s, récupération 3/3 et API saine. Le profil conserve donc la configuration réellement prouvée : eager, sans MTP, chunk 1024 et statique 0,88. Le seuil vLLM/DFlash2 de 2048 n'est pas généralisé à SGLang. [SGLang #36550](https://github.com/sgl-project/sglang/issues/36550) reproduit en plus un crash au premier token de décode après de longs prefills lorsque les graphes sont actifs. Les profils 384K/512K restent explicitement non sûrs et requièrent `--allow-unsafe-profile`.

DFlash2 peut faire progresser le décode bien au-delà de MTP sur code et sorties structurées, mais il ne répare pas la capacité longue. Le runtime de base ne contient pas le hook GLM requis ; `prepare-dflash2.sh` construit donc une image dérivée depuis la tête officielle SGLang #36708, réapplique les six correctifs SM121, puis télécharge le drafter épinglé sur les deux nœuds. Première mesure du 28 août : 37,2 tok/s mono-flux à 128K, soit +29 % vs MTP5 avec un TTFT intact ; la concurrence, l'acceptation par classe et l'égalité des sorties restent à valider avant promotion (voir [BENCHMARKS.md](docs/BENCHMARKS.md)). Les commandes et le repli FlashInfer sont dans [docs/DFLASH2.md](docs/DFLASH2.md).

Pour dépasser les ~29 tok/s mono-flux sans changer les poids ni la politique d'échantillonnage, `128k-mtp-ep1` isole un autre motif de communication MoE et `128k-mtp-compile` isole la compilation bs=1. Ils doivent être comparés séparément au profil MTP5 de référence ; une combinaison n'est justifiée que si chaque ablation gagne seule.

Validez chaque palier avec le bench en mode concurrence avant de l'adopter (voir section Benchmark). Les leviers supplémentaires — EP=1 contre EP=2, torch.compile, profondeur MTP, admission et fusion des rails RoCE — sont décrits avec leur risque qualité et leur protocole de mesure dans [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md). La requantification du `lm_head` est explicitement exclue de la trajectoire sans nouveau compromis qualité.

## Connexion depuis OpenCode

Deux configurations prêtes à adapter sont fournies :

- [examples/opencode.json](examples/opencode.json) pour OpenCode stable ;
- [examples/opencode-v2.jsonc](examples/opencode-v2.jsonc) pour OpenCode v2.

Les limites de contexte des deux exemples (`196608` tokens) sont calibrées pour
le profil long contexte par défaut `256k-dflash2-eager` (pool KV ≈ 210K tokens,
200 012 tokens froids validés) et laissent la place à la sortie. Si vous lancez
un autre profil, ajustez `limit.context` : `245760` pour `256k` (240 008 tokens
validés), `32768` pour `32k`.

Remplacez `10.10.10.1` par l'IP du head. Sans authentification SGLang, une valeur factice suffit au client :

```bash
export GLM53_API_KEY=local
```

Si `API_KEY` est définie dans `.env.glm53`, exportez la même valeur côté OpenCode.

`API_HOST=0.0.0.0` conserve l'écoute sur toutes les interfaces. Définissez
`API_ADVERTISE_HOST` sur l'adresse de management du head pour que les commandes
de statut et de démarrage affichent l'URL destinée aux clients, sans modifier le
fabric RoCE privé.

Par défaut, SGLang écoute sur `0.0.0.0` afin que le worker puisse observer la readiness du head. Le port `8888/tcp` est donc exposé sur les interfaces de la machine : configurez une clé API et/ou un filtrage réseau avant toute exposition hors du cluster de confiance.

## Benchmark

Le benchmark intégré mesure le TTFT, la durée totale et le débit de décodage en streaming. Le contenu généré n'est pas conservé, seulement son hash :

Si l'API est protégée, exportez d'abord `API_KEY` avec la valeur configurée dans
`.env.glm53`.

```bash
./bench-glm53.py --runs 3
```

Pour simuler des sous-agents et mesurer le comportement en charge, gardez N requêtes en vol simultanées. Le résumé affiche alors le TTFT p99 et le débit agrégé (goodput) :

```bash
./bench-glm53.py --runs 3 --concurrency 4
```

Le benchmark court ne valide pas la capacité long-contexte. Pour construire un vrai prompt calibré par le tokenizer SGLang, placer trois aiguilles à 5/50/95 %, vider le radix cache et vérifier que l'API survit au premier token de décode :

```bash
./bench-long-context.py \
  --allow-unsafe-profile \
  --target-tokens 480000 \
  --cold \
  --label 512k-mtp-cp
```

Le résultat JSON sépare le nombre de tokens demandé, le nombre réellement envoyé après le chat template, le TTFT/préfill, le débit de décode, la réussite des trois récupérations et la santé de l'API après la requête. Une limite de contexte n'est considérée validée que si ce test froid passe ; un simple `max_model_len` annoncé par `/v1/models` ne suffit pas.

Pour comparer une seconde API compatible OpenAI avec les mêmes prompts :

```bash
export ZAI_API_KEY='...'

./bench-glm53.py \
  --runs 3 \
  --compare-base-url 'https://API-OFFICIELLE/v1' \
  --compare-model 'glm-5.3-flash'
```

Les résultats sont écrits dans `results/`, ignoré par Git. Une comparaison de qualité sérieuse doit conserver les mêmes prompts, températures, budgets de tokens et tâches agentiques des deux côtés. Les mesures marquantes sont archivées dans [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

Pour mesurer la **consommation électrique** d'un bench (watts, énergie,
baseline idle), enveloppez-le avec
[`bench-power.py`](docs/POWER.md) :

```bash
./bench-power.py --label c6 -- python3 bench-glm53.py --runs 3 --concurrency 6
```

## Lane EXL3/vLLM (branche `dev`, répertoire `vllm-exl3/`)

Seconde lane produit : les poids **EXL3/TR3 4bpw** servis par **vLLM** via la
recette [MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
(MIT), vendorée puis durcie. **Validée sur le cluster le 2026-08-29** :
990 007 tokens froids validés au needle-exact (fenêtre 1M complète),
66.7 tok/s structured à l'acceptation 0.959, prefix-cache 4.1× sur les
follow-ups, poids ≈ FP8 officiel (KLD 0.0246 contre 0.0605 pour le NVFP4 à
taille égale) — au prix d'un TTFT batché moins bon (p99 16.7 s à C4 avec la
politique `skip`). Journal complet avec artefacts :
[BENCHMARKS.md](docs/BENCHMARKS.md) (source unique des deux lanes).

**Comparaison mesurée** (détails et artefacts dans le journal) :

| | Lane SGLang (défaut) | Lane EXL3 |
|---|---|---|
| Poids | NVFP4 communautaire | EXL3/TR3 4bpw (miroir octet-pour-octet) |
| Fidélité (KLD vs officiel) | 0.0605 nats | **0.0246 nats ≈ FP8 officiel, pour 54 % des octets** |
| Runtime | SGLang (6 correctifs SM121 audités) | vLLM + overlay 22 fichiers (sparse-MLA NoPE, MoE EXL3 fusionné, kernel CUDA E2 fat-expert) |
| Contexte validé | 240k froid (pool ≈ 210k) | **1M** (pool 1.75M tokens à util 0.87) |
| Décode mono | 37.2 tok/s (DFlash2, prompts standard) | 66.7 structured / 25.2 prose (DFlash2 k=7) |
| TTFT en concurrence | **0.7–1.0 s @ C4** | 6.3–6.6 s @ C4 (politique prefill-skip) |

**Quelle lane pour quoi :**

| Besoin | Lane |
|---|---|
| sous-agents OpenCode en rafales, TTFT interactif | **SGLang/NVFP4** (p99 0.7 s à C4) |
| contexte > 210k (jusqu'à 1M validé) | **EXL3/vLLM** |
| fidélité de poids maximale (code, raisonnement) | **EXL3/vLLM** |
| conversations multi-tours longues (réutilisation prefix) | **EXL3/vLLM** (4.1× sur les follow-ups) |

**Démarrage (2× Spark)** — depuis `vllm-exl3/` ; nécessite SSH sans mot de
passe head→worker, docker sans sudo sur les deux nœuds, ~180 GiB libres par
nœud. Les réglages propres au kit (noms de NIC, indices GID, repli mémoire)
sont documentés dans `vllm-exl3/.env.example` — lisez le bloc RoCE avant le
premier boot.

```bash
cd vllm-exl3
cp .env.example .env        # éditer HEAD_IP / WORKER_IP / NIC / GID
./download.sh               # optionnel : ~164 GiB dans le cache HF du head
./start.sh                  # tire l'image GHCR publique, synchronise, lance TP=2
```

> **Sync amont 2026-09-02** (`e25aa70`) : la lane embarque le kernel E2
> fat-expert prefill ([PR77 amont](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/77),
> **+20–21 % de prefill froid** 8k/100k/300k, décode inchangé), `MAX_NUM_BATCHED_TOKENS=7168`,
> `DFLASH_DRAFT_TP=2`, le garde-fou de config numérique, le recipe stamp
> (rebuild auto quand `Dockerfile`/`overlay/` changent après un git pull), les
> correctifs kpool-tail-slotmap et template (Reasoning Effort inconditionnel →
> prefix-cache), et `GLM53_INDEXER_WORKSPACE=rightsize` en opt-in (~+26 % de KV).
> L'image GHCR `:exl3` publique date du 2026-08-28 : le **premier** `./start.sh`
> après ce sync détectera l'écart de stamp et **reconstruira l'image localement**
> (couche CUDA incluse — comptez longtemps ; `SKIP_BUILD=1` pour rester sur
> GHCR avec `EXL3_FAT_KERNEL=0`). Nos correctifs locaux (#31 cache-reset) sont
> conservés et passent dans le hash du stamp.

API : `http://<HEAD_IP>:8888/v1`, modèle `GLM-5.3-Flash-EXL3`. Au premier run,
`.env.example` est copié vers `.env` ; l'export du shell bat `.env`
(`SPEC_METHOD=mtp ./start.sh restart` repasse la spéculation en MTP k=2).
Penser à `chat_template_kwargs: {"enable_thinking": false}` pour les petits
budgets de tokens.

**À ne pas faire sur cette lane** : backend `--moe-backend marlin`, poids
NVFP4, KV bf16 ou NVFP4 (`fp8_ds_mla` est la seule voie sparse-MLA sur SM12x) ;
épingler `TRITON_ATTN` pour le drafter (effondrement d'acceptation,
[EXL3-QUALITY.md](docs/EXL3-QUALITY.md)) ; baisser `MAX_MODEL_LEN` pour
« libérer » du KV ; monter `MAX_NUM_BATCHED_TOKENS` pour « réparer » le prefix
caching ; détruire les caches HF, requantizer, `docker rm` des caches ;
changer TP, pins CX7 ou `USE_HOST_NCCL` sans replomber NCCL.

**Interdictions amont toujours valables** : ne jamais combiner `tools` avec
`response_format`/`guided_json` dans une requête, et attendre des stalls de
prefill (10–40 s) quand de gros contextes arrivent concurremment — voir
[EXL3-KNOWN-ISSUES.md](docs/EXL3-KNOWN-ISSUES.md) avant toute charge agentique.

**Outils de la lane** (tout est exécuté depuis `vllm-exl3/`) : les quatre
protocoles de bench (`bench_decode.py`, `bench_prefix_cache.py`,
`bench_long_context.py`, `../bench-glm53.py`), le soak tool-calling issue #10
(`tests/soak_tool_calls.py`), l'A/B C4 scheduler (`scripts/c4-chunk-ab.sh` +
`tests/compare_c4.py`), le grader d'A/B qualité
(`tests/grade_ab_quality.py` + `tests/ab_quality_prompts.jsonl`), la sonde de
soak quotidien (`scripts/soak-day.sh`, protocole
[EXL3-SOAK.md](docs/EXL3-SOAK.md)) et la suite no-GPU `tests/run-local.sh`.

Par défaut, cette lane reste SGLang/NVFP4 tant que la lane EXL3 n'a pas
passé sa checklist de promotion (comparaison C4
`GLM53_MIXED_PREFILL_CHUNK=256`, validation tool-calling sous charge, soak
agentique multi-jours, A/B qualité contre l'API officielle) — checklist dans
[BENCHMARKS.md](docs/BENCHMARKS.md). Les deux
lanes partagent les outils génériques (`bench-glm53.py`) mais aucun état :
arrêtez une lane avant de démarrer l'autre.

## Reproductibilité et garde-fous

- révisions du modèle et de la source BF16 épinglées ;
- image SGLang ARM64/CUDA 13 corrigée pour SM121 et épinglée par digest ;
- contrôle de dérive upstream avant toute préparation ;
- snapshot local obligatoire au boot, sans téléchargement implicite ;
- validation de l'architecture `glm5_next` et de la configuration ModelOpt NVFP4 g16 ;
- validation exacte de 120 shards, 113 074 entrées et 37 152 poids experts quantifiés ;
- vérification des hashes du tokenizer, du processor, de la generation config et du chat template ;
- refus des fichiers exécutables inattendus dans le dépôt modèle ;
- limites mémoire `mem_limit == memswap_limit == 120g` et surveillance de `MemAvailable` ;
- démarrage worker-first et arrêt coordonné en cas d'échec.

Le runtime vLLM officiel du jour de sortie n'est pas utilisé : son chemin NoPE produit une sortie incorrecte sur GB10. L'image SGLang retenue contient six correctifs audités et dispose d'une validation TP=2 sur deux DGX Spark.

La provenance, les révisions et les invariants du checkpoint sont détaillés dans [docs/AUDIT.md](docs/AUDIT.md).

## Tests de la recette

La suite locale vérifie la syntaxe, tous les profils Compose, le validateur fail-closed, l'orchestration worker-first et les clients de benchmark court/long simulés :

```bash
./tests/run-local.sh
```

Ces tests ne téléchargent pas le checkpoint complet et ne remplacent pas un démarrage sur deux GB10.

## Documentation complémentaire

- [audit du checkpoint](docs/AUDIT.md) ;
- [optimisation : vitesse, concurrence et contexte](docs/OPTIMIZATION.md) ;
- [mesure de la consommation électrique](docs/POWER.md) ;
- [audit et trajectoire DFlash2](docs/DFLASH2.md) ;
- [runbook expérimental 256K/384K/512K/MTP/CUDA graphs/DFlash2](docs/EXPERIMENT-RUNBOOK.md) ;
- [configuration et diagnostic RoCE](docs/NETWORK.md) ;
- [historique des benchmarks](docs/BENCHMARKS.md) — source unique des deux lanes ;
- [qualité EXL3 : KLD, acceptation DFlash2](docs/EXL3-QUALITY.md) ;
- [issues amont connues de la lane EXL3](docs/EXL3-KNOWN-ISSUES.md) ;
- [protocole et journal du soak multi-jours EXL3](docs/EXL3-SOAK.md) ;
- [guide de dépannage](docs/TROUBLESHOOTING.md) ;
- [crédits et inspirations](CREDITS.md).

## Sources principales

- [checkpoint NVFP4 LibertAIDAI](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) ;
- [source officielle BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16) ;
- [runtime SGLang SM121 audité](https://github.com/0xSero/glm-5.3-flash-sglang-sm121) ;
- [documentation multi-nœud SGLang](https://docs.sglang.ai/backend/pd_disaggregation.html) ;
- [recette 2× Spark de MiaAI-Lab](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) ;
- [recette Hy3 NVFP4 sur 2× GB10](https://huggingface.co/LibertAIDAI/Hy3-NVFP4/tree/main/deploy) ;
- [recette EXL3 vLLM de MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) (lane `vllm-exl3/`) ;
- [checkpoint EXL3/TR3 4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) et [son miroir public](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw) ;
- [drafter DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) (CC BY-NC-ND 4.0) ;
- [kernels ExLlamaV3](https://github.com/turboderp-org/exllamav3) (format EXL3).

Le code de cette recette est distribué sous licence MIT. Les poids, l'image SGLang et leurs licences restent des artefacts externes distincts.
