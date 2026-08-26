# Audit du checkpoint et du correctif de template

Audit effectué le 26 août 2026 vers 21:11 CEST.

## Conclusion

Le checkpoint NVFP4 épinglé par cette recipe contient le template corrigé. Il n'est pas nécessaire de requantifier les poids : le correctif upstream et sa synchronisation NVFP4 modifient chacun un seul fichier, `chat_template.jinja`.

Un cache NVFP4 pris avant la révision `11d7321…` doit en revanche récupérer le nouveau snapshot. Hugging Face réutilise les blobs de poids inchangés ; le pin force surtout la récupération du bon fichier de template et crée le snapshot correspondant à la bonne révision.

## Chronologie vérifiée

| Dépôt | Révision | Date UTC | Événement |
|---|---|---|---|
| [`zai-org/GLM-5.3-Flash-BF16`](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/commit/b1967181a3917ae70a437f4884748f6b8e3a1f4d) | `b1967181a3917ae70a437f4884748f6b8e3a1f4d` | 16:30:51 | `update template`, 1 fichier |
| [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash/commit/3f1971b7b5f7a528c9c4ef6212c8785298a8c24a) | `3f1971b7b5f7a528c9c4ef6212c8785298a8c24a` | 16:31:15 | HEAD officiel post-correctif |
| [`LibertAIDAI/GLM-5.3-Flash-NVFP4`](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4/commit/11d73216cd636238e82e1d77fe1042ffab36e7fa) | `11d73216cd636238e82e1d77fe1042ffab36e7fa` | 18:07:05 | synchronisation explicite du template, 1 fichier |

Titre exact du commit quant : `Sync chat_template.jinja with upstream (2026-08-26 16:31Z): fixes multimodal image/video/audio token emission`.

Les trois `chat_template.jinja` donnent :

```text
SHA-256 34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891
```

## Checkpoint NVFP4 épinglé

- 120 shards ;
- payload tensor indexé : `194644803576` octets, soit 181,28 GiB ;
- 113 074 entrées dans le weight map ;
- 37 152 poids routed-expert NVFP4 ;
- 37 152 `weight_scale` et 37 152 `weight_scale_2` ;
- 1 618 entrées non-expert ;
- architecture `Glm5NextForConditionalGeneration`, type `glm5_next` ;
- ModelOpt 0.45.0, NVFP4 weight-only, 4 bits float, groupe 16 ;
- attention, shared experts, routers, vision, embeddings, `lm_head`, normes et MTP gardés hors du chemin de quantification expert.

Les métadonnées auditées sont dans [metadata/checkpoint-manifest.json](../metadata/checkpoint-manifest.json). Le validateur recalcule les invariants et les hashes sur chaque nœud avant le boot.

## Fichiers de tokenizer comparés à la source BF16

| Fichier | SHA-256 |
|---|---|
| `tokenizer.json` | `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d` |
| `tokenizer_config.json` | `98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc` |
| `chat_template.jinja` | `34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891` |
| `generation_config.json` | `230c30609ecbbb9e6583bedde8e7bdda0c6eb8fe5fad0eaeb3d1b293d751cb4f` |
| `processor_config.json` | `aae38374c94b08cc9b0547c6e64f05b951bd9735cea571c6988f5ed552bed3ed` |

## Gate de dérive

`./check-upstream-glm53.sh` compare les HEADs actuels, les trois templates et le digest Docker au manifeste. Toute dérive bloque la préparation : elle doit d'abord être lue, comprise, puis intégrée avec un nouveau pin et un manifeste mis à jour.
