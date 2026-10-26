# Credits

Pour le socle commun et la lane `sglang-nvfp4/` :

- Z.ai / `zai-org` pour GLM-5.3-Flash et le checkpoint BF16 officiel.
- LibertAIDAI pour la quantification ModelOpt NVFP4, la provenance détaillée et les validations GB10 publiées.
- SGLang pour le serveur compatible OpenAI et le support `glm5_next`.
- 0xSero pour l'image SGLang SM121 reproductible, ses six correctifs audités et les résultats d'acceptation TP=2/TP=4.
- MiaAI-Lab pour le pattern de lancement worker-first, TP=2, multiprocessing, RoCE/NCCL/Gloo sur deux DGX Spark.
- La recipe `LibertAIDAI/Hy3-NVFP4/deploy` pour les garde-fous de mémoire unifiée GB10.

Pour la lane `vllm-exl3/` :

- MiaAI-Lab pour la recette vLLM/EXL3 vendorée (MIT) : overlay sparse-MLA NoPE, méthode EXL3 fused MoE, intégration DFlash2 k=7, patches prefix-cache et xgrammar, et ses contributeurs d'issues (PR #26 GID par rang, issue #22 robustesse de bring-up).
- brandonmusic pour le checkpoint EXL3/TR3 4bpw (ShapleyMCG License 1.0) et Mia-AiLab pour son miroir public byte-identical.
- turboderp et l'équipe ExLlamaV3 pour le format EXL3 et les kernels `exl3_moe`.
- IncoAI pour le drafter GLM-5.3-Flash-DFlash2 (CC BY-NC-ND 4.0, recherche/évaluation).
- malaiwah pour le panel KLD indépendant (weights-vs-official), socle de la décision produit EXL3.
- vLLM et les auteurs des backports #52805/#53046 pour le runtime speculative/grammaire.
- drowzeys pour la recette d'abliteration « altered Dealign » (L15–43 + MTP 45, ancres L0–14 et L44 ; MIT) et dealignai pour les tenseurs `o_proj` donneurs de `GLM-5.3-Flash-UNCENSORED-NVFP4` (MIT), utilisés par `ABLIT=1`.

Cette recipe réimplémente et adapte ces idées à GLM-5.3-Flash ; les artefacts upstream conservent leurs propres licences.

## Politique de licence du vendor `vllm-exl3/`

Le code vendoré dans `vllm-exl3/` provient de
[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
au commit `e25aa70` (2026-09-02), publié sous **MIT** (`vllm-exl3/LICENSE`).
L'amont a été relicencié en **AGPL-3.0** le 2026-09-07 (commit `cf0f430`).
Conséquences pour ce dépôt, distribué sous MIT :

- aucun code amont postérieur à `cf0f430^` n'entre dans `vllm-exl3/` sans
  décision de licence explicite (relicencier ce répertoire en AGPL, ou
  réimplémenter indépendamment) ;
- les rapports d'issues, mesures et choix de configuration publiés en amont
  restent utilisables comme faits ; seul le code est concerné ;
- la recette successeur
  [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
  est sous Apache-2.0, compatible avec ce dépôt sous réserve de conserver son
  avis de licence et ses notices.

Avant tout sync, `vllm-exl3/scripts/check-vendor-license.sh <ref>` refuse (code 1) une ref dont la licence n'est pas MIT.

## Ajouts 2026-10 — durcissement agentique et noyaux

Cette passe ajoute du code et des idées provenant de kits communautaires
Apache-2.0, avec attribution explicite :

- **[Reederey87/glm53-flash-exl3-2x-dgx-spark](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark)**
  (Apache-2.0) : overlays de prefix-cache agentique (`patch_fine_grained_apc.py`,
  `patch_apc_per_group_retention.py`, `patch_apc_no_store.py`,
  `patch_prefix_cache_sparse_miss_metric.py`), variante 3-stage `cp.async` du
  fat-GEMM (`exl3_fat_gemm.cu`), et les tests hôtes correspondants. Les fichiers
  portés gardent leur en-tête SPDX Apache-2.0 et sont listés dans
  [`vllm-exl3/NOTICE`](vllm-exl3/NOTICE).
- **[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)**
  (Apache-2.0) : idées et mesures servant de référence pour les prochains
  chantiers (4-bit dense, KV compact, spill tiers, graphes drafter). Aucun code
  TensorFold n'est vendoré dans cette passe ; les fichiers qui en seraient
  dérivés devront conserver l'avis Apache-2.0 et être ajoutés à `NOTICE`.
- **MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks** : le vendor reste au snapshot
  MIT `e25aa70`. Les mesures et rapports publiés après le passage AGPL
  (`cf0f430`, 2026-09-07) restent utilisables comme faits ; le code post-AGPL
  n'entre pas dans ce dépôt sans décision de licence explicite.
