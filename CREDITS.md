# Credits

- Z.ai / `zai-org` pour GLM-5.3-Flash et le checkpoint BF16 officiel.
- LibertAIDAI pour la quantification ModelOpt NVFP4, la provenance détaillée et les validations GB10 publiées.
- SGLang pour le serveur compatible OpenAI et le support `glm5_next`.
- 0xSero pour l'image SGLang SM121 reproductible, ses six correctifs audités et les résultats d'acceptation TP=2/TP=4.
- MiaAI-Lab pour le pattern de lancement worker-first, TP=2, multiprocessing, RoCE/NCCL/Gloo sur deux DGX Spark.
- La recipe `LibertAIDAI/Hy3-NVFP4/deploy` pour les garde-fous de mémoire unifiée GB10.

Pour la lane `vllm-exl3/` (branche `dev`) :

- MiaAI-Lab pour la recette vLLM/EXL3 vendorée (MIT) : overlay sparse-MLA NoPE, méthode EXL3 fused MoE, intégration DFlash2 k=7, patches prefix-cache et xgrammar, et ses contributeurs d'issues (PR #26 GID par rang, issue #22 robustesse de bring-up).
- brandonmusic pour le checkpoint EXL3/TR3 4bpw (ShapleyMCG License 1.0) et Mia-AiLab pour son miroir public byte-identical.
- turboderp et l'équipe ExLlamaV3 pour le format EXL3 et les kernels `exl3_moe`.
- IncoAI pour le drafter GLM-5.3-Flash-DFlash2 (CC BY-NC-ND 4.0, recherche/évaluation).
- malaiwah pour le panel KLD indépendant (weights-vs-official), socle de la décision produit EXL3.
- vLLM et les auteurs des backports #52805/#53046 pour le runtime speculative/grammaire.

Cette recipe réimplémente et adapte ces idées à GLM-5.3-Flash ; les artefacts upstream conservent leurs propres licences.
