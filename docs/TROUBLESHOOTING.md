# Dépannage

Commencez toujours par conserver les logs des deux rangs :

```bash
./logs-glm53.sh --profile 32k --node both --tail 500
```

## Ancienne image vLLM ou ancien fichier `.env.glm53`

La recette utilise désormais le runtime SGLang SM121 audité. Si la validation mentionne `GLM53_VLLM_IMAGE`, `11d7321…` ou l'image `vllm/vllm-openai:glm53-flash-arm64-cu130`, recréez votre configuration à partir du nouvel exemple et recopiez uniquement vos valeurs cluster :

```bash
cp .env.glm53 .env.glm53.before-sglang
cp .env.glm53.example .env.glm53
$EDITOR .env.glm53
```

Ne recopiez pas les anciens pins de modèle ou de runtime.

## Route fabric incorrecte

Une erreur telle que :

```text
expected dev enp1s0f0np0 src 192.168.100.10
```

signifie que Linux choisirait une autre interface ou une autre IP source pour une socket non liée. Sur un fabric Spark à deux liens déjà validé, NCCL/Gloo/TP peuvent néanmoins utiliser correctement `.10/.11` grâce à leur liaison explicite à `enp1s0f0np0`. La recette avertit sans bloquer lorsque `STRICT_FABRIC_ROUTE=0`. Consultez [NETWORK.md](NETWORK.md).

## GID vide ou `tr: erreur de lecture: Argument invalide`

Une ancienne version de la recette lisait un index GID fixe. Supprimez ces variables de `.env.glm53` :

```text
NCCL_IB_GID_INDEX
WORKER_NCCL_IB_GID_INDEX
```

Le runtime utilise NCCL ≥ 2.21 et sélectionne automatiquement le GID RoCE v2. Renseignez plutôt `NCCL_IB_ADDR_RANGE`.

## Code 137, garde mémoire déclenché ou nœud presque figé

- arrêtez les autres modèles et conteneurs sur les deux machines ;
- désactivez le swap pendant la mise en service si possible ;
- restez sur `32k`, une requête et sans MTP ;
- inspectez `.glm53-guard-head.log` et le fichier worker équivalent ;
- conservez la valeur du profil au lieu de la surcharger : 0,90 pour le socle
  32K–128K, 0,88 pour le profil de fiabilité `256k` ;
- utilisez `32k-eager` pour exclure la capture CUDA graphs.

Le garde arrête uniquement le conteneur portant les labels Compose exacts de cette recette.

Le garde mémoire est exclusivement un garde de **démarrage**. Il attend une
tokenisation réelle (pas seulement l'ouverture de `/v1/models`) et doit être
désarmé avant le smoke test et avant toute charge réelle. Après la readiness,
`status-glm53.sh` retourne une erreur si le garde du head est encore actif.

## `SIGTERM` pendant un préfill long alors que le serveur était prêt

Si les blocs de préfill commencent normalement, puis qu'un `SIGTERM received`
apparaît sans erreur CUDA préalable, inspectez `.glm53-guard-head.log`. Une
ancienne version sondait le head via `MASTER_ADDR`; sur certains routages locaux,
la sonde ne voyait jamais l'API et le garde restait armé. Lorsque le préfill
faisait ensuite passer `MemAvailable` sous son seuil, il arrêtait proprement le
conteneur, ce qui produisait `Remote end closed connection` côté benchmark et
des `TCPStore Broken pipe` secondaires sur le worker.

La recette courante sonde le head via `127.0.0.1`, attend l'arrêt effectif des
deux gardes avant d'accepter du trafic et refuse le benchmark long si le garde
local est encore vivant. Mettez la recette à jour, arrêtez les deux rangs, puis
refaites un démarrage complet du profil.

## Mémoire KV insuffisante

SGLang réserve les poids, le cache KV et l'état hybride KDA/Mamba dans `MEM_FRACTION_STATIC`. L'erreur `Not enough GPU memory for hybrid (mamba/linear-attention) state cache` avec une valeur négative de `total_rest_memory` indique que cette fraction est trop basse, pas trop haute. Le socle TP=2 32K–128K utilise `MEM_FRACTION_STATIC=0.90`. Le profil `256k` utilise volontairement 0,88 pour rendre environ 2,56 Gio à l'exécution dynamique sur une machine de 128 Gio ; si ce profil ne passe pas le démarrage complet, n'augmentez pas la valeur puis n'envoyez pas 240K : conservez les logs et revenez à 128K.

## Blocage NCCL ou Gloo

Causes probables :

- mauvais HCA ou nom d'interface ;
- route vers le pair utilisant le réseau de management ;
- IP configurée absente de l'interface Gloo ;
- plage `NCCL_IB_ADDR_RANGE` ne contenant pas l'IP fabric ;
- ancien `NCCL_IB_GID_INDEX` encore défini.

Relancez `./doctor-glm53.sh` jusqu'à ce que les deux routes, les HCA et les GID dynamiques soient validés.

## Le serveur tente de télécharger pendant le boot

Arrêtez-le. `HF_HUB_OFFLINE=1` et `TRANSFORMERS_OFFLINE=1` doivent rester actifs. Exécutez `./prepare-glm53.sh` jusqu'à validation des 120 shards sur les deux nœuds.

## Le modèle répond mais `start-glm53.sh` affiche encore `Still loading`

Les anciennes versions utilisaient `/health` pour le garde mémoire et le healthcheck Docker. Sur ce runtime, cet endpoint peut déclencher une passe synthétique de 64 tokens ; avec une seule requête active, les sondes répétées pouvaient retarder `/v1/models` indéfiniment. La recette sonde désormais `/v1/models`, vérifie l'identifiant exact, puis exige une réponse tokenizer valide avant d'annoncer le serveur prêt.

## Le benchmark long-contexte reçoit `HTTP 503` sur `/tokenize`

Si le corps HTML mentionne **Squid**, la requête loopback a été envoyée au proxy :
ce n'est pas une réponse de SGLang. La recette courante impose un accès direct
pour `127.0.0.1`, `localhost` et les sondes du fabric ; `export no_proxy=...`
n'est plus nécessaire.

Un 503 JSON ou texte provenant réellement de SGLang signifie que le frontal
répond encore, mais que le moteur d'inférence charge, s'arrête ou n'est plus
vivant. Vérifiez l'état des deux rangs avant de relancer :

```bash
./status-glm53.sh 256k
./logs-glm53.sh --profile 256k --node both --tail 200
```

Le client long-contexte réessaie désormais ces erreurs transitoires pendant
60 secondes et affiche séparément l'état de `/v1/models`. Le démarrage n'annonce
plus « prêt » avant une réponse valide du tokenizer. Un journal contenant
`SIGTERM received`, puis `SystemExit: 0`, décrit un arrêt externe propre et non
un crash de kernel. Relancez `./start-glm53.sh 256k`, attendez son message final,
confirmez `tokenizer ready` avec la commande de statut, puis lancez le benchmark.
N'exécutez pas `stop-glm53.sh`,
`docker compose stop/down` ou un arrêt de conteneur pendant le test.

## Scheduler `exit code -9` pendant un préfill 240K

`-9` est un `SIGKILL`, pas une exception Python ou CUDA récupérable. Sur le run
du 27 août, le préfill `256k-mtp` a progressé jusqu'à environ 164K tokens, s'est
tu pendant plus de huit minutes, puis le scheduler a été tué. Avec la limite
conteneur de 120 Gio et 128 Gio de mémoire CPU/GPU unifiée, cette signature est
compatible avec une mise à mort par pression mémoire/cgroup ; le log applicatif
seul ne permet pas de distinguer le cgroup d'un OOM killer hôte.

La correction n'est pas de relancer le même profil. Utilisez `256k`, qui retire
les graphes et MTP, passe le chunk 4096 → 1024 et la fraction statique 0,90 →
0,88. `bench-long-context.py` inspecte le profil effectif et refuse désormais
automatiquement `256k-mtp` au-dessus de 128K. Un override explicite existe pour
le diagnostic, mais il rétablit volontairement le risque de crash.

## `Permission denied` dans `/cache/huggingface`

Le chemin `/cache/huggingface` est le cache hôte monté dans le conteneur. Si un ancien conteneur lancé en root a créé le dossier du modèle, rendez uniquement ce dossier et son dossier de verrous au compte qui exécute la recette :

```bash
sudo chown -R "$(id -u):$(id -g)" \
  "$HOME/.cache/huggingface/hub/models--LibertAIDAI--GLM-5.3-Flash-NVFP4" \
  "$HOME/.cache/huggingface/hub/.locks/models--LibertAIDAI--GLM-5.3-Flash-NVFP4"
```

Un dossier de verrous absent est sans gravité. Ne supprimez pas le cache : les blobs déjà présents sont réutilisables.

## Proxy GHCR sur le worker

`prepare` réutilise maintenant l'image auditée lorsque sa référence exacte par digest est déjà locale et ne contacte alors pas le registre. Si seul le head peut tirer l'image, transférez-la avec `docker save | ssh docker load`, puis vérifiez la référence exacte avec `docker image inspect`. Une image tirée uniquement par digest peut apparaître `<untagged>` dans `docker image ls` sans être absente.

## Checkpoint ou config rejeté

La révision exigée est `f4aa9ef9b180d608b924fade8983dca18b9bcdf7`. Elle ne modifie aucun poids par rapport à `11d7321…` ; Hugging Face réutilise donc les blobs existants dans le même cache.

Si `check-upstream-glm53.sh` signale un nouveau HEAD, ne remplacez pas simplement le SHA. Le diff, l'index, la configuration et les hashes doivent d'abord être audités.

## Sortie qui répète le prompt

C'est la signature connue du chemin NoPE de l'image vLLM officielle sur GB10. Vérifiez que les deux rangs utilisent exactement :

```text
ghcr.io/0xsero/glm-5.3-flash-sglang-sm121
@sha256:f9ac60ba4071f8acd64f0f3c074aca308f6d659405fee46fc8031489a1e8b19b
```

Deux images différentes entre les rangs donnent souvent un boot trompeur ou une sortie incorrecte.

## Erreur DSA, FlashInfer ou module quantifié

Ne remplacez pas les backends du profil 32K. Le chemin audité exige :

```text
DSA prefill/decode = flashinfer_sparse_mla
MoE = flashinfer_cutlass
KV = fp8_e4m3
shared-expert fusion = disabled
```

Capturez les deux logs avant toute modification.

## MTP ne charge pas

Revenez à `32k`. Le profil `32k-mtp` dépend de correctifs supplémentaires pour le draft NEXTN quantifié et ne doit être essayé qu'après une validation complète sans spéculation.

## Échec de capture CUDA graphs à 256k

Le profil `256k-graphs` réactive la capture des graphes à 262 144 tokens avec `CUDA_GRAPH_MAX_BS=1`. Si le boot échoue sur une erreur mémoire pendant la capture (« CUDA out of memory » côté capture/replay) :

- baissez `MEM_FRACTION_STATIC` à 0,88 dans une copie du profil (le validateur accepte jusqu'à 0,92 mais émet un avertissement au-delà de 0,90) ;
- sinon retombez sur le profil `256k` eager, qui reste la sonde de référence.

Un OOM pendant la capture n'est pas silencieux : le launcher collecte les logs et arrête les deux rangs.

## Le préfill >262k finit puis le serveur tombe au premier token

C'est distinct d'un OOM de capture. [SGLang #36550](https://github.com/sgl-project/sglang/issues/36550)
documente un abort dans le replay du graphe de décode après un préfill froid
qui franchit 262 144 tokens. Un benchmark court avec une limite serveur 512k ne
détecte pas ce défaut.

- confirmez avec `./bench-long-context.py --target-tokens 300000 --cold` ;
- capturez immédiatement les logs des deux rangs ;
- utilisez `512k-mtp-eager` pour désactiver le replay tout en gardant MTP ;
- testez ensuite `512k-mtp-cp`, d'abord à 300k, puis 400k et 480k. Le préfill
  CP=2 garde environ la moitié de la séquence sur chaque rang, mais ce chemin
  reste expérimental tant que les trois aiguilles et la santé post-requête ne
  sont pas validées sur les deux GB10.

## Combinaisons de knobs refusées

Le validateur fail-closed refuse deux combinaisons instables plutôt que de laisser SGLang échouer au boot :

- `ENABLE_MIXED_CHUNK=1` avec `MTP_NUM_TOKENS>0` : le prefill mélangé n'est pas supporté sous spéculation. Gardez `ENABLE_MIXED_CHUNK` pour les profils sans MTP ;
- `EP_SIZE` autre que 1 ou 2 : la recette fixe `--tp-size 2`.

Toute autre valeur doit passer par une modification consciente du validateur et une nouvelle ligne d'audit.

## Le boot dépasse une heure

Un MoE de 320B peut charger lentement, mais une heure sans API doit être traitée comme un échec. Le launcher collecte les logs et arrête les rangs. Cherchez le dernier progrès de chargement, une compilation JIT, un OOM ou une attente NCCL.
