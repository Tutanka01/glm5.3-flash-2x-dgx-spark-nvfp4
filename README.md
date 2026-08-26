# GLM-5.3-Flash NVFP4 sur 2× DGX Spark / PGX (GB10)

Recipe reproductible pour servir `LibertAIDAI/GLM-5.3-Flash-NVFP4` sur deux machines GB10 en tensor parallel `TP=2`, via vLLM et une API compatible OpenAI.

Le profil par défaut cherche d'abord un boot fiable : ARM64/CUDA 13, backend MoE Marlin, eager mode, KV FP8, contexte 32K, une seule séquence, aucun MTP. Les profils plus ambitieux ne sont activés qu'après un premier `GLM53_OK`.

> État au 26 août 2026 : la recipe, ses pins et ses validations sont prêts. Le checkpoint est extrêmement récent et tiers ; sa qualité n'est pas encore validée indépendamment. Cette machine de développement ne possède pas les deux GB10, donc le premier boot matériel reste à exécuter sur le cluster.

## Alerte « retélécharger les poids » : prise en compte

L'annonce concernait une correction du `chat_template.jinja`, pas une modification numérique des poids. L'ordre exact est rassurant :

- source BF16 corrigée à `2026-08-26 16:30:51Z`, révision `b196718…` ;
- dépôt officiel corrigé à `16:31:15Z`, révision `3f1971b…` ;
- quant NVFP4 resynchronisé ensuite à `18:07:05Z`, révision `11d7321…` ;
- le commit NVFP4 change un seul fichier, `chat_template.jinja` ;
- les templates officiel, BF16 et NVFP4 ont tous le SHA-256 `34d5ee66…3891`.

La recipe épingle donc précisément le quant **post-correctif** `11d73216cd636238e82e1d77fe1042ffab36e7fa`. `check-upstream-glm53.sh` échoue si un dépôt ou le tag d'image bouge, et la validation locale refuse un ancien template. Détails : [docs/AUDIT.md](docs/AUDIT.md).

## Architecture

```text
head (rank 0)                              worker (rank 1, headless)
start-glm53.sh ─────── SSH, worker-first ───────► vLLM
      │                                           │
      └────────── TP=2 / NCCL+Gloo / RoCE ────────┘
      │
      └── http://HEAD:8888/v1  (API sur rank 0 uniquement)
```

Chaque nœud conserve une copie complète du checkpoint (~181,3 GiB sur disque), puis vLLM répartit les tenseurs entre les deux rangs (~90,64 GiB de payload indexé par nœud en répartition idéale).

## Prérequis

- 2× NVIDIA DGX Spark, Lenovo ThinkStation PGX ou autre machine GB10 (`aarch64`, `sm_121`) ;
- driver CUDA 13 compatible, Docker et plugin `docker compose` v2 ;
- Python 3 et `curl` sur le head ;
- lien RoCE connu fonctionnel entre les machines ;
- SSH sans mot de passe du head vers le worker ;
- au moins 205 GiB libres dans le cache Hugging Face de chaque nœud ;
- idéalement swap désactivé pendant le bring-up.

Le réseau est volontairement explicite. Reprends les valeurs `WORKER_HOST`, `MASTER_ADDR`, `VLLM_HOST_IP`, interfaces, HCA et GID d'une recipe TP=2 déjà fonctionnelle sur ces deux machines.

## Démarrage rapide

Sur le head :

```bash
cp .env.glm53.example .env.glm53
$EDITOR .env.glm53
chmod +x ./*.sh ./scripts/*.sh
```

Valide d'abord uniquement la configuration :

```bash
./validate-glm53.sh --config-only
./doctor-glm53.sh
```

Vérifie ensuite que les pins distants n'ont pas changé, tire l'image dédiée sur les deux nœuds, télécharge le snapshot complet sur chacun et valide les 120 shards :

```bash
./check-upstream-glm53.sh
./prepare-glm53.sh
```

Le téléchargement est séquentiel par défaut pour ne pas saturer le même accès Internet. Pour deux accès indépendants :

```bash
PREPARE_PARALLEL=1 ./prepare-glm53.sh
```

Premier boot :

```bash
./start-glm53.sh 32k
```

Le launcher démarre le worker en premier, lance un garde mémoire sur chaque nœud, attend jusqu'à une heure, puis exige une réponse finale contenant `GLM53_OK`. Il arrête les deux rangs automatiquement si l'un tombe avant que l'API soit prête.

Contrôles courants :

```bash
./status-glm53.sh 32k
./logs-glm53.sh --profile 32k --node both --tail 300
./smoke-glm53.sh --profile 32k --tools
./stop-glm53.sh --profile 32k
```

## Profils

| Profil | Contexte | Séquences | Backend | Eager | MTP | Usage |
|---|---:|---:|---|---:|---:|---|
| `32k` | 32K | 1 | Marlin | oui | non | premier boot |
| `64k` | 64K | 1 | Marlin | oui | non | deuxième étape |
| `128k` | 128K | 1 | Marlin | oui | non | expérimental |
| `256k` | 256K | 1 | Marlin | oui | non | test de plafond mémoire |
| `32k-mtp` | 32K | 1 | Marlin | oui | 5 | après validation sans MTP |
| `32k-native` | 32K | 1 | auto/NVFP4 | oui | non | sonde kernels natifs sm_121 |

Toujours arrêter avant de changer de profil :

```bash
./stop-glm53.sh --profile 32k
./start-glm53.sh 64k
```

Si `32k-native` retourne `cudaErrorNoKernelImageForDevice`, reviens au profil `32k`. Ne retire pas `--enforce-eager` avec Marlin sur GB10 : les retours existants montrent des blocages pendant la capture CUDA graphs.

## OpenCode

Deux exemples sont fournis :

- [examples/opencode.json](examples/opencode.json) pour la configuration OpenCode stable ;
- [examples/opencode-v2.jsonc](examples/opencode-v2.jsonc) pour OpenCode v2.

Remplace `10.10.10.1` par l'IP du head. Si l'API n'a pas de clé, une valeur factice suffit au client :

```bash
export GLM53_API_KEY=local
```

Si `VLLM_API_KEY` est définie dans `.env.glm53`, exporte la même valeur côté OpenCode.

Le bind par défaut est `0.0.0.0` afin que le garde du worker voie la readiness du head. Il expose donc le port sur les interfaces du head : définis `VLLM_API_KEY` et/ou filtre `8888/tcp` au niveau réseau avant toute exposition au-delà du cluster de confiance.

## Benchmark local contre API officielle

Le harness standard-library mesure TTFT, durée et débit de décodage en streaming, sans enregistrer le texte généré :

```bash
./bench-glm53.py --runs 3
```

Comparaison avec une autre API compatible OpenAI :

```bash
export ZAI_API_KEY='...'
./bench-glm53.py \
  --runs 3 \
  --compare-base-url 'https://API-OFFICIELLE/v1' \
  --compare-model 'glm-5.3-flash'
```

Les résultats sont écrits sous `results/` et ignorés par Git. Ne publie pas de conclusion qualité avant d'avoir utilisé les mêmes prompts, températures, budgets et tâches agentiques des deux côtés.

## Garde-fous inclus

- modèle, source BF16, image ARM64 et digest tous épinglés ;
- vérification en ligne des HEADs avant préparation ;
- snapshot offline obligatoire au boot : aucun téléchargement sauvage de 181 GiB ;
- validation de `glm5_next`, ModelOpt 0.45, NVFP4 g16 et chemins BF16 protégés ;
- validation exacte de 120 shards, 113 074 entrées et 37 152 poids experts quantifiés ;
- hashes du tokenizer, processor, generation config et template corrigé ;
- refus de tout fichier exécutable inattendu dans le dépôt modèle ;
- `mem_limit == memswap_limit == 112g` et garde `MemAvailable` ;
- worker-first, IP vLLM par rang, interfaces NCCL/TP/Gloo par nœud ;
- diagnostic et arrêt coordonné si le boot échoue.

Voir [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) pour les signatures d'erreur attendues et [docs/NETWORK.md](docs/NETWORK.md) pour la vérification RoCE.

La suite locale (syntaxe, six profils Compose, validation fail-closed, orchestration worker-first et clients API simulés) se lance avec :

```bash
./tests/run-local.sh
```

## Sources principales

- [checkpoint NVFP4 LibertAIDAI](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) ;
- [source officielle BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16) ;
- [support GLM-5.3 vLLM PR #53906](https://github.com/vllm-project/vllm/pull/53906) ;
- [recipe 2× Spark MiaAI-Lab](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) ;
- [recipe Hy3 NVFP4 2× GB10](https://huggingface.co/LibertAIDAI/Hy3-NVFP4/tree/main/deploy).

Les poids, l'image vLLM et leurs licences restent des artefacts externes distincts.
