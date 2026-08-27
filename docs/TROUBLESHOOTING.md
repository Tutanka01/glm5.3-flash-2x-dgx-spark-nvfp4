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

signifie que Linux rejoint le pair par une autre interface ou une autre IP source. Mettez `HEAD_FABRIC_IP`, `WORKER_FABRIC_IP`, les interfaces et les HCA en accord avec la route réelle. Consultez [NETWORK.md](NETWORK.md).

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
- baissez `MEM_FRACTION_STATIC` par pas de `0.01` si nécessaire ;
- utilisez `32k-eager` pour exclure la capture CUDA graphs.

Le garde arrête uniquement le conteneur portant les labels Compose exacts de cette recette.

## Mémoire KV insuffisante

SGLang réserve les poids et le cache KV dans `MEM_FRACTION_STATIC`. Libérez d'abord la mémoire système, puis augmentez prudemment cette valeur sans dépasser `0.90`. Le cap conteneur de 112g reste une barrière supplémentaire.

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

## Le boot dépasse une heure

Un MoE de 320B peut charger lentement, mais une heure sans API doit être traitée comme un échec. Le launcher collecte les logs et arrête les rangs. Cherchez le dernier progrès de chargement, une compilation JIT, un OOM ou une attente NCCL.
