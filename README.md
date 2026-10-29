# GLM-5.3-Flash sur 2× DGX Spark

Recettes reproductibles et mesurées pour servir **GLM-5.3-Flash** (MoE 320B)
sur deux machines **NVIDIA GB10** (DGX Spark / ThinkStation PGX) en tensor
parallel `TP=2` sur RoCE, avec une API compatible OpenAI sur le port `8888`.

Le dépôt propose **deux lanes** : deux façons indépendantes de servir le même
modèle, chacune avec sa propre configuration, ses scripts et ses tests. Elles
partagent les outils de benchmark et un journal de mesures unique. Une seule
lane tourne à la fois sur le cluster.

| | [**SGLang / NVFP4**](sglang-nvfp4/README.md) | [**vLLM / EXL3**](vllm-exl3/README.md) |
|---|---|---|
| Statut | **lane par défaut**, validée | validée sur cluster, en promotion (2 soaks restants) |
| Poids | NVFP4 communautaire, 181 GiB | EXL3/TR3 4bpw, ~164 GiB |
| Fidélité (KLD vs officiel) | 0,0605 nats | **0,0246 nats ≈ FP8 officiel** |
| Contexte validé à froid | 240K | **1M** (990K tokens, 3/3 aiguilles) |
| Décode mono-flux | 37,2 tok/s (DFlash2) | 66,7 structured / 25,2 prose (DFlash2 k=7) |
| TTFT sous concurrence C4 | **0,7–1,0 s** | p99 1,2 s avec `GLM53_MIXED_PREFILL_CHUNK=256` |
| Particularités | 27 profils (32K → 256K, MTP, DFlash2) | prefix-cache 4,1×, variante abliterated `ABLIT=1` |

Protocoles et artefacts de chaque chiffre : [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Quelle lane choisir

| Besoin | Lane |
|---|---|
| premier déploiement, sous-agents OpenCode en rafales | **SGLang / NVFP4** |
| contexte au-delà de 210K (jusqu'à 1M) | **vLLM / EXL3** |
| fidélité de poids maximale (code, raisonnement) | **vLLM / EXL3** |
| longues conversations multi-tours (réutilisation du prefix) | **vLLM / EXL3** |

## Démarrage rapide

Prérequis communs : 2× GB10 sous Linux ARM64, Docker et Docker Compose v2,
SSH sans mot de passe du head vers le worker, un lien RoCE fonctionnel entre
les deux, 180 à 205 GiB libres par nœud selon la lane. Toutes les commandes se lancent
sur le **head** ; les scripts synchronisent le worker eux-mêmes.

```bash
git clone https://github.com/Tutanka01/glm5.3-flash-2x-dgx-spark-nvfp4.git
cd glm5.3-flash-2x-dgx-spark-nvfp4
```

**SGLang / NVFP4** : parcours guidé complet dans [sglang-nvfp4/README.md](sglang-nvfp4/README.md).

```bash
cd sglang-nvfp4
cp .env.glm53.example .env.glm53 && chmod 600 .env.glm53 && $EDITOR .env.glm53
./validate-glm53.sh --config-only && ./doctor-glm53.sh
./prepare-glm53.sh && ./start-glm53.sh 32k
```

**vLLM / EXL3** : détails dans [vllm-exl3/README.md](vllm-exl3/README.md).

```bash
cd vllm-exl3
cp .env.example .env && $EDITOR .env
./start.sh
```

Pour passer d'une lane à l'autre, arrêtez d'abord celle qui tourne
(`sglang-nvfp4/stop-glm53.sh --profile <profil>` ou `vllm-exl3/stop.sh`).

### Variante abliterated (sans refus)

Disponible uniquement sur la lane **vLLM / EXL3**, désactivée par défaut. Un
seul interrupteur, sur le head :

```bash
cd vllm-exl3
ABLIT=1 ./start.sh restart        # ou ABLIT=1 dans .env pour la rendre permanente
```

Au premier démarrage, le head télécharge ~2,6 GiB de tenseurs `o_proj` du
donneur épinglé (reprise possible), les copie sur le worker, et les deux rangs
les substituent au chargement : aucun poids n'est réécrit sur disque.
`ABLIT=0 ./start.sh restart` revient au modèle d'origine.

- Vérification : `/v1/models` expose `GLM-5.3-Flash-EXL3-Ablit` (l'id
  d'origine reste un alias) et les logs affichent `ablit: ON`.
- Cache KV insuffisant au démarrage (`max seq len … KV cache is needed`) :
  ajoutez `GLM53_INDEXER_WORKSPACE=rightsize`, ou baissez `MAX_MODEL_LEN`
  sous le plafond indiqué par vLLM. Ne dépassez pas `GPU_MEM_UTIL=0.87`.
- Après un crash, `./stop.sh` et vérifiez qu'aucun conteneur ne reste sur les
  deux nœuds (`docker ps -a`) avant de relancer : sinon NCCL peut échouer
  avec `ibv_reg_mr … Cannot allocate memory`.
- Le modèle répond à tout : gardez la clé API et le filtrage réseau.

Recette, garanties et réglages `ABLIT_*` :
[vllm-exl3/README.md](vllm-exl3/README.md#variante-abliterated-ablit1) et
[REFERENCE.md](vllm-exl3/REFERENCE.md#abliteration-ablit1).

## Organisation du dépôt

```text
.
├── sglang-nvfp4/   lane SGLang : scripts *-glm53.sh, profiles/, scripts/, dflash2/, tests/
├── vllm-exl3/      lane vLLM : start.sh, overlay/, ablit/, scripts/, tests/
├── tools/          outils communs : bench-glm53.py, bench-power.py, prompts
├── docs/
│   ├── BENCHMARKS.md   journal unique des mesures des deux lanes
│   ├── POWER.md        mesure de la consommation électrique
│   ├── sglang/         audit, réseau, DFlash2, optimisation, dépannage
│   ├── exl3/           qualité, cache agentique, issues connues, soak
│   └── img/            figures (courbes de puissance)
├── run-tests.sh    tous les tests sans GPU
└── CREDITS.md
```

Chaque lane écrit ses résultats bruts dans son propre `results/` (ignoré
par Git) et se teste avec son propre `tests/run-local.sh`. À la racine,
`./run-tests.sh` lance tout.

## Mettre à jour un clone d'avant la réorganisation

Jusqu'au 2026-10, la lane SGLang vivait à la racine du dépôt. Sur le head :

```bash
./stop-glm53.sh --profile <profil actif>   # avant le pull, sinon le PID du garde est perdu
git pull
mv .env.glm53 sglang-nvfp4/
[ -d results ] && mkdir -p sglang-nvfp4/results && mv results/* sglang-nvfp4/results/ && rmdir results
```

Si `.env.glm53` est resté à la racine, les scripts l'utilisent encore, avec
un avertissement. Les noms des scripts et des profils n'ont pas changé : seul
le dossier de travail change (`cd sglang-nvfp4`). Le dossier `WORKER_DIR` du
worker reste valide tel quel.

> Le nom du dépôt (`…-nvfp4`) est historique : il date d'avant la lane EXL3.

## Licences

Le code de ce dépôt est sous [MIT](LICENSE). Le code vendoré dans
`vllm-exl3/` reste sous la licence MIT de son amont (voir
[CREDITS.md](CREDITS.md#politique-de-licence-du-vendor-vllm-exl3)). Les poids,
les images Docker et le drafter DFlash2 (CC BY-NC-ND 4.0) sont des artefacts
externes, chacun sous sa propre licence. Crédits et inspirations :
[CREDITS.md](CREDITS.md).
