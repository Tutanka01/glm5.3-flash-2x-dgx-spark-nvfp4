# Troubleshooting

Commence toujours par conserver les logs des deux rangs :

```bash
./logs-glm53.sh --profile 32k --node both --tail 500
```

## `glm5_next` inconnu ou architecture non supportée

Le mauvais conteneur est utilisé. La recipe exige l'image dédiée ARM64/CUDA 13 et son digest audité. Exécute `./prepare-glm53.sh`; ne remplace pas l'image par `latest`.

## `cudaErrorNoKernelImageForDevice` / erreur FP4 MoE sur `sm_121`

Tu as probablement lancé `32k-native`. Arrête les deux rangs puis reviens au fallback connu :

```bash
./stop-glm53.sh --profile 32k-native
./start-glm53.sh 32k
```

Le profil `32k` force `--moe-backend marlin --enforce-eager`.

## Code 137, garde mémoire déclenché ou nœud presque figé

- arrête les autres modèles/containers sur les deux machines ;
- vérifie que `mem_limit` et `memswap_limit` valent 112g dans le rendu Compose ;
- désactive le swap pendant le bring-up si possible ;
- garde `32k`, `MAX_NUM_SEQS=1`, eager et sans MTP ;
- inspecte `.glm53-guard-head.log` et le fichier équivalent worker ;
- si nécessaire, baisse `GPU_MEMORY_UTILIZATION` par pas de 0,01, sans descendre trop bas : vLLM doit encore réserver un pool KV non nul.

Le garde arrête uniquement le container portant les labels Compose exacts de cette recipe.

## `No available memory for the cache blocks`

Le modèle et le runtime consomment déjà le budget ciblé. Libère de la mémoire système. Si le nœud est propre, augmente avec prudence `GPU_MEMORY_UTILIZATION` jusqu'à 0,90 ; le cap container 112g reste la dernière barrière.

## `Gloo connectFullMesh Connection refused`

Vérifie `VLLM_HOST_IP` séparément sur chaque rang et fixe `GLOO_SOCKET_IFNAME` sur l'interface RoCE. Lance `./doctor-glm53.sh` et consulte [NETWORK.md](NETWORK.md).

## NCCL bloque pendant l'initialisation

Les causes les plus probables sont : mauvais `NCCL_IB_HCA`, GID non RoCE v2, index différent entre les nœuds, IP non portée par l'interface ou route qui repasse par le réseau de management. Reprends les valeurs d'une recipe TP=2 déjà validée sur le même fabric.

## Le serveur télécharge pendant le boot

Arrête-le. `HF_HUB_OFFLINE=1` et `TRANSFORMERS_OFFLINE=1` doivent rester actifs. Exécute `./prepare-glm53.sh` jusqu'à ce que la validation des 120 shards réussisse sur les deux nœuds.

## Le template ou le checkpoint est rejeté

Un cache antérieur au correctif peut être présent. La révision exigée est `11d7321…`. Relance `./prepare-glm53.sh` : les blobs de poids inchangés sont réutilisés et le snapshot post-correctif est créé.

Si `check-upstream-glm53.sh` signale un nouveau HEAD, n'édite pas simplement le SHA dans `.env.glm53`. Audite le diff, actualise le manifeste et recalcule les hashes.

## Sortie incohérente ou `NotImplementedError` attention

Reste sur le profil 32K conservateur et l'image dédiée. Capture les deux logs, le résultat du smoke et l'environnement. Des incidents similaires ont existé avec GLM-5.2 sur GB10/FlashInfer ; ils ne prouvent pas que GLM-5.3 a le même bug, mais justifient de ne publier aucune mesure qualité avant comparaison avec l'API officielle.

## Le boot dépasse une heure

Un 320B MoE peut charger lentement, mais une heure sans API doit être traitée comme un échec. Le launcher récupère les logs et arrête les rangs. Cherche le dernier progrès de chargement, une compilation JIT, un OOM ou une attente NCCL.

