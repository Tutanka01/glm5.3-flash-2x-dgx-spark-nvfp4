# Credits

- Z.ai / `zai-org` pour GLM-5.3-Flash et le checkpoint BF16 officiel.
- LibertAIDAI pour la quantification ModelOpt NVFP4, la provenance détaillée et la synchronisation rapide du template corrigé.
- vLLM pour l'image modèle ARM64/CUDA 13 et le support `glm5_next` en cours d'intégration.
- MiaAI-Lab pour le pattern de lancement worker-first, TP=2, multiprocessing, RoCE/NCCL/Gloo sur deux DGX Spark.
- La recipe `LibertAIDAI/Hy3-NVFP4/deploy` pour les garde-fous GB10 validés autour de Marlin, eager mode et la limite mémoire container.

Cette recipe réimplémente et adapte ces idées à GLM-5.3-Flash ; les artefacts upstream conservent leurs propres licences.
