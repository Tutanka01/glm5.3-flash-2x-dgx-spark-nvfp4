# Audit du checkpoint et du runtime

Dernier audit : 27 août 2026 à 09:39 CEST.

## Conclusion

La révision NVFP4 épinglée est `f4aa9ef9b180d608b924fade8983dca18b9bcdf7`. Entre l'ancien pin `11d7321…` et cette révision, seuls `README.md` et `config.json` ont changé :

- les 120 fichiers de poids Git/LFS sont identiques ;
- `model.safetensors.index.json` est identique ;
- `chat_template.jinja` est identique ;
- tous les champs JSON de `config.json` sont identiques hors de `quantization_config.ignore` ;
- 11 noms de modules fusionnés ont été ajoutés à cette liste pour le chargement SGLang.

Il n'est donc pas nécessaire de requantifier ou de retélécharger les poids déjà présents dans le même cache Hugging Face. Un nouveau snapshot Git est créé en réutilisant les blobs inchangés.

## Historique vérifié

| Dépôt | Révision | Date UTC | Événement |
|---|---|---|---|
| [`zai-org/GLM-5.3-Flash-BF16`](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/commit/b1967181a3917ae70a437f4884748f6b8e3a1f4d) | `b196718…` | 2026-08-26 16:30:51 | mise à jour du template |
| [`LibertAIDAI/GLM-5.3-Flash-NVFP4`](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4/commit/11d73216cd636238e82e1d77fe1042ffab36e7fa) | `11d7321…` | 2026-08-26 18:07:05 | synchronisation du template |
| [`LibertAIDAI/GLM-5.3-Flash-NVFP4`](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4/commit/cf5434c00bf69bd0e6b58420c9636999472a2291) | `cf5434c…` | 2026-08-27 07:24:27 | ajout des noms fusionnés à la liste `ignore` |
| [`LibertAIDAI/GLM-5.3-Flash-NVFP4`](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4/commit/f4aa9ef9b180d608b924fade8983dca18b9bcdf7) | `f4aa9ef…` | 2026-08-27 07:24:39 | nouvelle model card et HEAD audité |

Les trois templates — distribution officielle, source BF16 et quant NVFP4 — ont le même hash :

```text
SHA-256 34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891
```

## Checkpoint NVFP4

- 120 shards ;
- payload tensor indexé : `194644803576` octets, soit 181,28 GiB ;
- 113 074 entrées dans le weight map ;
- 37 152 poids routed-expert NVFP4 ;
- 37 152 `weight_scale` et 37 152 `weight_scale_2` ;
- 1 618 entrées non-expert ;
- architecture `Glm5NextForConditionalGeneration`, type `glm5_next` ;
- ModelOpt 0.45.0, NVFP4 weight-only, 4 bits float, groupe 16 ;
- 43 chemins exclus de la quantification, dont les noms fusionnés utilisés par SGLang.

## Hashes verrouillés

| Fichier | SHA-256 |
|---|---|
| `config.json` | `5db46f44956e4a8a0cc8ed54b6d77bf99dd7c1ec90c58975d1952560768513d5` |
| `model.safetensors.index.json` | `f3a4c40897e00fab0de0380b05b66279bc341233cc14fa71a80bab2b683e3b7b` |
| `tokenizer.json` | `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d` |
| `tokenizer_config.json` | `98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc` |
| `chat_template.jinja` | `34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891` |
| `generation_config.json` | `230c30609ecbbb9e6583bedde8e7bdda0c6eb8fe5fad0eaeb3d1b293d751cb4f` |
| `processor_config.json` | `aae38374c94b08cc9b0547c6e64f05b951bd9735cea571c6988f5ed552bed3ed` |

## Choix du runtime

La nouvelle model card signale que l'image vLLM officielle du jour de sortie produit une sortie incorrecte sur GB10 : le modèle est NoPE (`qk_rope_head_dim=0`) alors que le chemin MLA `sm_121` concerné attend une dimension positionnelle.

La recette utilise désormais l'image SGLang suivante :

```text
ghcr.io/0xsero/glm-5.3-flash-sglang-sm121
@sha256:f9ac60ba4071f8acd64f0f3c074aca308f6d659405fee46fc8031489a1e8b19b
```

Audit du runtime :

- source : [`0xSero/glm-5.3-flash-sglang-sm121`](https://github.com/0xSero/glm-5.3-flash-sglang-sm121) ;
- commit source : `dfb1bb7e45c20058d37df7a39cceda45a9d216a8` ;
- manifeste OCI Linux ARM64 : `sha256:7aff51ea7050480dc47137055b5201b73e23e7803d9439233742cab65e3e5609` ;
- base SGLang épinglée par digest ;
- six fichiers de correctifs hashés ;
- validation publiée sur 2× GB10, TP=2/EP=2, avec texte cohérent, tool calling et multimodal.

L'image contient un template personnalisé historique, mais cette recette ne le passe pas à SGLang. Le serveur charge directement le `chat_template.jinja` hashé du snapshot `f4aa9ef…`.

## Gate de dérive

`./check-upstream-glm53.sh` vérifie :

- les trois HEADs Hugging Face ;
- les trois templates ;
- l'existence du commit source SGLang audité ;
- le manifeste OCI Linux ARM64 de l'image épinglée.

Toute dérive bloque la préparation. Les invariants et hashes locaux sont ensuite recalculés sur chaque nœud avant le démarrage.
