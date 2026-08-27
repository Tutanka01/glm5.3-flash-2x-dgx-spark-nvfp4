# Credits

- Z.ai / `zai-org` pour GLM-5.3-Flash et le checkpoint BF16 officiel.
- LibertAIDAI pour la quantification ModelOpt NVFP4, la provenance détaillée et les validations GB10 publiées.
- SGLang pour le serveur compatible OpenAI et le support `glm5_next`.
- 0xSero pour l'image SGLang SM121 reproductible, ses six correctifs audités et les résultats d'acceptation TP=2/TP=4.
- MiaAI-Lab pour le pattern de lancement worker-first, TP=2, multiprocessing, RoCE/NCCL/Gloo sur deux DGX Spark.
- La recipe `LibertAIDAI/Hy3-NVFP4/deploy` pour les garde-fous de mémoire unifiée GB10.

Cette recipe réimplémente et adapte ces idées à GLM-5.3-Flash ; les artefacts upstream conservent leurs propres licences.
