# GLM-5.3-Flash NVFP4 sur 2× DGX Spark / PGX

Recette de déploiement reproductible pour servir `LibertAIDAI/GLM-5.3-Flash-NVFP4` sur deux machines GB10 en tensor parallel (`TP=2`), avec SGLang et une API compatible OpenAI.

L'objectif initial est volontairement simple et vérifiable : charger le modèle sur les deux nœuds, exposer une API sur le head et obtenir une réponse cohérente. La configuration par défaut privilégie donc la fiabilité avant la performance : runtime ARM64/CUDA 13 corrigé pour `sm_121`, MoE `flashinfer_cutlass`, attention DSA corrigée, cache KV FP8, contexte 32K, une seule requête et aucun MTP.

> **Statut :** le runtime épinglé a produit des réponses cohérentes en TP=2 sur deux GB10, mais l'intégration de cette recette doit encore être validée sur votre fabric. Le checkpoint reste communautaire ; sa qualité doit être comparée indépendamment à l'API officielle.

## Architecture

```text
head (rank 0)                              worker (rank 1, headless)
start-glm53.sh ─────── SSH, worker-first ───────► SGLang
      │                                           │
      └────────── TP=2 / NCCL + Gloo / RoCE ──────┘
      │
      └── http://HEAD:8888/v1  (API sur le head uniquement)
```

Chaque nœud conserve une copie complète du checkpoint, soit environ 181,3 GiB sur disque. SGLang répartit ensuite les tenseurs entre les deux rangs, pour une charge idéale d'environ 90,64 GiB de poids indexés par nœud.

Le démarrage est coordonné depuis le head : synchronisation de la recette, validation locale du checkpoint sur les deux machines, lancement du worker, lancement du head, attente de l'API puis smoke test automatique.

## Prérequis

- 2× NVIDIA DGX Spark, Lenovo ThinkStation PGX ou autre machine GB10 (`aarch64`, `sm_121`) ;
- driver compatible CUDA 13, Docker et plugin `docker compose` v2 ;
- Python 3 et `curl` sur le head ;
- lien RoCE fonctionnel entre les deux machines ;
- connexion SSH non interactive du head vers le worker ;
- au moins 205 GiB libres dans le cache Hugging Face de chaque nœud ;
- swap désactivé de préférence pendant la mise en service initiale.

La recette ne devine volontairement pas la topologie réseau. Reprenez les IP, interfaces, HCA et index GID d'une configuration TP=2 déjà fonctionnelle sur ces deux machines — par exemple votre déploiement DeepSeek existant.

## Installation

Toutes les commandes suivantes s'exécutent sur le head.

### Migration depuis la première version vLLM

Si votre `.env.glm53` contient encore `GLM53_VLLM_IMAGE` ou `NCCL_IB_GID_INDEX`, repartez du nouvel exemple :

```bash
cp .env.glm53 .env.glm53.before-sglang
cp .env.glm53.example .env.glm53
$EDITOR .env.glm53
```

Recopiez uniquement vos paramètres SSH, chemins de cache et valeurs réseau confirmées. Conservez les nouveaux pins `MODEL_REVISION` et `GLM53_RUNTIME_IMAGE`, ne recopiez pas l'ancien index GID, puis passez directement à l'étape 2.

### 1. Configurer le cluster

Pour une installation neuve :

```bash
cp .env.glm53.example .env.glm53
$EDITOR .env.glm53
```

Renseignez au minimum :

- `WORKER_HOST` et `WORKER_DIR` ;
- `MASTER_ADDR`, `HEAD_FABRIC_IP` et `WORKER_FABRIC_IP` ;
- les interfaces `NCCL`, `TP` et `GLOO` ;
- le HCA RoCE et `NCCL_IB_ADDR_RANGE`.

Les identifiants du modèle, sa révision et l'image SGLang sont déjà épinglés. Ne les modifiez qu'après un nouvel audit. Avec NCCL ≥ 2.21, ne définissez pas `NCCL_IB_GID_INDEX` : le GID RoCE v2 est sélectionné dynamiquement.

### 2. Valider l'environnement

```bash
./validate-glm53.sh --config-only
./doctor-glm53.sh
```

Le doctor contrôle notamment l'architecture ARM64, le GPU, Docker, la mémoire disponible, le cache, les interfaces réseau, le HCA et le GID sur les deux nœuds.

### 3. Préparer l'image et le checkpoint

```bash
./prepare-glm53.sh
```

Cette commande :

1. vérifie que les révisions distantes et le digest de l'image correspondent toujours à l'audit ;
2. synchronise la recette vers le worker ;
3. tire l'image SGLang épinglée sur les deux nœuds ;
4. télécharge le snapshot complet sur chacun ;
5. valide les 120 shards et les métadonnées de quantification.

Le téléchargement est séquentiel par défaut afin de ménager un accès Internet partagé. Pour deux accès indépendants :

```bash
PREPARE_PARALLEL=1 ./prepare-glm53.sh
```

### 4. Effectuer le premier démarrage

```bash
./start-glm53.sh 32k
```

Le script démarre le worker en premier, active un garde mémoire sur chaque nœud et attend jusqu'à une heure que l'API réponde. Le démarrage est considéré comme réussi uniquement si le smoke test obtient `GLM53_OK`.

L'API est alors disponible à l'adresse :

```text
http://<HEAD_FABRIC_IP>:8888/v1
```

Si l'un des deux rangs tombe avant la readiness, les logs sont collectés et les deux conteneurs sont arrêtés automatiquement.

## Exploitation courante

```bash
# État des deux rangs et de l'API
./status-glm53.sh 32k

# Logs récents du head et du worker
./logs-glm53.sh --profile 32k --node both --tail 300

# Test chat + tool calling
./smoke-glm53.sh --profile 32k --tools

# Arrêt coordonné
./stop-glm53.sh --profile 32k
```

Pour suivre les logs en continu, sélectionnez un seul nœud :

```bash
./logs-glm53.sh --profile 32k --node head --tail 300 --follow
```

## Profils de lancement

| Profil | Contexte | Requêtes | MoE | Graphes | MTP | Usage recommandé |
|---|---:|---:|---|---|---:|---|
| `32k` | 32 768 | 1 | FlashInfer CUTLASS | oui | non | premier démarrage |
| `32k-batch4` | 32 768 | 4 | FlashInfer CUTLASS | oui | non | sous-agents OpenCode |
| `32k-batch8` | 32 768 | 8 | FlashInfer CUTLASS | oui | non | concurrence élevée, expérimental |
| `64k` | 65 536 | 1 | FlashInfer CUTLASS | oui | non | deuxième étape |
| `128k` | 131 072 | 1 | FlashInfer CUTLASS | oui | non | long contexte mono-requête |
| `128k-batch4` | 131 072 | 4 | FlashInfer CUTLASS | oui | non | sous-agents sur longs contextes |
| `128k-batch8` | 131 072 | 8 | FlashInfer CUTLASS | oui | non | longs contextes + forte concurrence |
| `256k` | 262 144 | 1 | FlashInfer CUTLASS | non | non | recherche de la limite mémoire |
| `32k-mtp` | 32 768 | 1 | FlashInfer CUTLASS | oui | 5 étapes | après validation sans MTP |
| `32k-eager` | 32 768 | 1 | FlashInfer CUTLASS | non | non | diagnostic sans CUDA graphs |

Les noms de profils sont des abréviations : `128k` correspond à `MAX_MODEL_LEN=131072`, soit 131 072 tokens (128 × 1024).

Arrêtez toujours le profil actif avant d'en charger un autre :

```bash
./stop-glm53.sh --profile 32k
./start-glm53.sh 32k-batch4
```

Progressez dans l'ordre `32k` → `64k` → `128k` → `256k`, puis testez MTP. Utilisez `32k-eager` uniquement pour isoler un problème de capture ou de replay CUDA graphs.

## Concurrence et sous-agents

Le profil `32k` n'accepte qu'une seule requête (`MAX_NUM_SEQS=1`) : c'est un choix de fiabilité pour le premier démarrage, pas une limite du modèle. Les profils `32k-batch4` et `32k-batch8` lèvent cette limite et permettent à OpenCode de lancer plusieurs sous-agents en parallèle.

À savoir en passant au batché :

- le débit de décode **par requête** baisse légèrement (la bande passante mémoire est partagée), mais le débit **agrégé** augmente nettement ;
- le radix cache de SGLang est actif : les requêtes qui partagent un préfixe identique (system prompt des sous-agents) réutilisent le KV et un préfill quasi gratuit ;
- si le garde mémoire se déclenche ou que le préfill échoue sur `32k-batch8`, revenez à `32k-batch4` ou remettez `MAX_NUM_BATCHED_TOKENS=4096`.

Les profils `128k-batch4` et `128k-batch8` combinent 131 072 tokens de contexte et la concurrence. Le pool KV est partagé entre les requêtes actives : si quatre conversations de 131k ne tiennent pas simultanément, SGLang met simplement les requêtes en excès en file d'attente au lieu de les rejeter. En usage agentique réel, peu de conversations remplissent tout le contexte, donc la concurrence utile est généralement supérieure au cas le pire.

Validez chaque palier avec le bench en mode concurrence avant de l'adopter (voir section Benchmark).

## Connexion depuis OpenCode

Deux configurations prêtes à adapter sont fournies :

- [examples/opencode.json](examples/opencode.json) pour OpenCode stable ;
- [examples/opencode-v2.jsonc](examples/opencode-v2.jsonc) pour OpenCode v2.

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

```bash
./bench-glm53.py --runs 3
```

Pour simuler des sous-agents et mesurer le comportement en charge, gardez N requêtes en vol simultanées. Le résumé affiche alors le TTFT p99 et le débit agrégé (goodput) :

```bash
./bench-glm53.py --runs 3 --concurrency 4
```

Pour comparer une seconde API compatible OpenAI avec les mêmes prompts :

```bash
export ZAI_API_KEY='...'

./bench-glm53.py \
  --runs 3 \
  --compare-base-url 'https://API-OFFICIELLE/v1' \
  --compare-model 'glm-5.3-flash'
```

Les résultats sont écrits dans `results/`, ignoré par Git. Une comparaison de qualité sérieuse doit conserver les mêmes prompts, températures, budgets de tokens et tâches agentiques des deux côtés.

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

La suite locale vérifie la syntaxe, les six profils Compose, le validateur fail-closed, l'orchestration worker-first et les clients API simulés :

```bash
./tests/run-local.sh
```

Ces tests ne téléchargent pas le checkpoint complet et ne remplacent pas un démarrage sur deux GB10.

## Documentation complémentaire

- [audit du checkpoint](docs/AUDIT.md) ;
- [configuration et diagnostic RoCE](docs/NETWORK.md) ;
- [guide de dépannage](docs/TROUBLESHOOTING.md) ;
- [crédits et inspirations](CREDITS.md).

## Sources principales

- [checkpoint NVFP4 LibertAIDAI](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) ;
- [source officielle BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16) ;
- [runtime SGLang SM121 audité](https://github.com/0xSero/glm-5.3-flash-sglang-sm121) ;
- [documentation multi-nœud SGLang](https://docs.sglang.ai/backend/pd_disaggregation.html) ;
- [recette 2× Spark de MiaAI-Lab](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) ;
- [recette Hy3 NVFP4 sur 2× GB10](https://huggingface.co/LibertAIDAI/Hy3-NVFP4/tree/main/deploy).

Le code de cette recette est distribué sous licence MIT. Les poids, l'image SGLang et leurs licences restent des artefacts externes distincts.
