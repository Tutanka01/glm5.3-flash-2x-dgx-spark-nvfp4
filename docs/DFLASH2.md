# DFlash2 : audit et trajectoire d'intégration

Audit réalisé le 28 août 2026 à partir du dépôt
[`tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark`](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark),
révision `9642d4f6628bb66e5fd03afe8e1c31bb2922c6c3`, du
[drafter officiel Inco AI](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
et du [support GLM DFlash dans SGLang #36708](https://github.com/sgl-project/sglang/pull/36708).
Les pins structurés sont conservés dans
[`metadata/dflash2-candidate.json`](../metadata/dflash2-candidate.json).

## Verdict

DFlash2 est la meilleure piste connue pour accélérer le décode mono-flux sans
modifier la distribution de sortie : le modèle cible vérifie les propositions.
L'image SGLang de base n'a pas ce hook. Le cookbook fournit maintenant une
image dérivée reproductible et quatre profils directement lançables ; ils sont
expérimentaux jusqu'au premier retour matériel sur vos deux GB10.

| Élément | Niveau de preuve | Décision |
|---|---|---|
| DFlash2 officiel | drafter 1B BF16, bloc 8, 7 drafts ; évaluation officielle sur 4× GB300 | algorithme crédible, matériel différent |
| Port 2× DGX Spark cité | vLLM patché, 46,9 tok/s C1 code chaud, 74,1 % d'acceptation ; C1–C6 sans échec avec KV bridé | excellente référence expérimentale, résultats auto-publiés à reproduire |
| Runtime de base du cookbook | SGLang SM121 épinglé avant le hook GLM requis | ne pas lui passer les flags DFlash |
| Image dérivée du cookbook | PR SGLang exacte `2d4b6ac…`, six correctifs SM121 réappliqués, adaptateur mHC vérifié à la construction | profils expérimentaux prêts |
| SGLang #36708 | hook GLM + contraction mHC, tests locaux ; fusionné dans une branche GLM, pas dans `main`, empilé sur #36507 encore ouverte | construire puis auditer une nouvelle image, ne pas patcher le conteneur en place |
| Licence du drafter | CC BY-NC-ND 4.0, recherche/évaluation | licence commerciale obligatoire hors évaluation |

Le dépôt externe est une intégration **vLLM**. Ses quatre patches touchent la
capture des états cachés, le registre du drafter et surtout le groupement KV
hybride propre à vLLM. Les copier dans SGLang serait techniquement incorrect.
La voie SGLang part de la révision exacte #36708 `2d4b6ac…`. Le Dockerfile
remet ensuite les six fichiers SM121 de `0xSero@dfb1bb7…` et applique
l'adaptateur GLM/mHC avec des ancres fail-closed. Il refuse la construction si
DFLASH générique, FA4, le worker DFlash ou le hook GLM manquent.

## Lancement prêt à tester

La préparation construit la même image sur les deux nœuds, valide le modèle
cible, puis télécharge et contrôle le drafter 1B épinglé :

```bash
./prepare-dflash2.sh 128k-dflash2
./start-glm53.sh 128k-dflash2
./bench-glm53.py --warmup-runs 1 --runs 3 --concurrency 1
./bench-glm53.py --warmup-runs 1 --runs 3 --concurrency 4
```

Pour chercher le sommet de débit agrégé observé autour de C5 dans le port
Spark/vLLM, chargez ensuite la voie graphes bs=8 et balayez sans changer les
prompts :

```bash
./stop-glm53.sh --profile 128k-dflash2
./start-glm53.sh 128k-dflash2-c8
for c in 1 2 4 5 6; do
  ./bench-glm53.py --warmup-runs 1 --runs 3 --concurrency "$c"
done
```

Si le boot échoue spécifiquement dans FlashAttention 4, gardez l'image et le
drafter déjà préparés et changez uniquement le backend draft :

```bash
./stop-glm53.sh --profile 128k-dflash2-c8
./start-glm53.sh 128k-dflash2-flashinfer
./bench-glm53.py --warmup-runs 1 --runs 3 --concurrency 1
```

Le profil de pression 240K est également prêt :

```bash
./stop-glm53.sh --profile 128k-dflash2-flashinfer
./start-glm53.sh 256k-dflash2-eager
./bench-long-context.py --allow-unsafe-profile \
  --target-tokens 240000 --cold --label 256k-dflash2-eager
```

Après un échec, lancez immédiatement
`./collect-glm53-report.sh --profile <profil>` avant tout arrêt. L'archive
contient les logs des deux rangs, leur état Docker, la mémoire, le GPU et les
événements kernel OOM sans exporter `API_KEY` ni l'environnement du conteneur.

## Ce que l'on récupère immédiatement

1. **Toujours chauffer avant de mesurer.** Les premiers appels compilent les
   kernels DFlash2 ; le port a observé environ 10 tok/s de pénalité sur une
   mesure C1 froide. `bench-glm53.py` effectue désormais un warmup écarté des
   statistiques (`--warmup-runs 1` par défaut).
2. **Mesurer par classe de prompt.** L'acceptation est meilleure sur code et
   sorties structurées que sur prose. Une moyenne unique peut masquer une
   régression. Conserver au minimum code, raisonnement et prose.
3. **Chercher le bon niveau de concurrence.** Le port TP2 atteint son débit
   agrégé maximal à C5 (56,2 tok/s) puis régresse à C6. DFlash2 cible surtout la
   latence mono-flux ; sans drafter peut rester meilleur pour du débit batché.
4. **Garder une réserve mémoire réelle.** Un premier pool KV plus agressif a
   été tué à C3 pendant trois prefills concurrents de 20K. Le profil publié
   sacrifie environ 118K tokens de pool pour survivre. Sur GB10, la mémoire
   unifiée disponible et les allocations transitoires comptent plus que la
   capacité KV nominale. Deuxième confirmation le 28 août : le c8 à statique
   0,92 n'a jamais atteint la readiness — le garde de démarrage du head a
   coupé le conteneur (MemAvailable 6032 MiB < plancher 6144 MiB) pendant la
   capture des graphes draft bs=8, avec 91,0 Go de poids + 4,1 Go de Mamba
   40 slots déjà posés. Le profil tourne désormais à 0,90, la réserve du
   `128k-dflash2-c4` validé.
5. **Ne pas généraliser le chunk entre moteurs.** Le port vLLM/DFlash2 rapporte
   des segfaults de warmup sous `index_topk=2048`, donc nos profils DFlash2
   expérimentaux restent à 2048. En revanche, SGLang sans DFlash2 a réellement
   réussi 240 008 tokens avec chunk 1024 ; le profil `256k` conserve cette
   valeur prouvée.
6. **Contrôler l'acceptation, pas seulement la vitesse.** Une mauvaise
   contraction mHC peut fonctionner tout en chutant vers 15 % d'acceptation.
   La promotion exige l'acceptation par classe de prompt et l'égalité des
   sorties déterministes avec le chemin cible seul.

Les commandes privilégiées du dépôt vLLM — `--kv-cache-memory`, groupement de
pages KV, vidage agressif du page cache, patches dans `site-packages` — ne sont
pas transposées. Notre recette conserve ses gardes mémoire et son image
immuable.

## Sas automatique de compatibilité

Le contrôle suivant ne télécharge rien et échoue avec l'image de base, ce qui
est attendu :

```bash
./scripts/check-dflash2-runtime.sh
```

Après `prepare-dflash2.sh`, contrôlez l'image construite et le snapshot :

```bash
set -a
source .env.glm53
set +a
./scripts/check-dflash2-runtime.sh \
  --image 'glm53-sglang-dflash2:2d4b6ac-sm121' \
  --draft-dir "$HF_CACHE/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/7d74cdd881ed7e32c31175984a67823127b66cfe"
```

Le sas vérifie dans le code de l'image : l'algorithme DFLASH générique, le hook
`set_dflash_layers_to_capture` sur GLM-5.3, la contraction mHC et le backend
d'attention draft FA4. Il vérifie aussi la géométrie du `config.json` du drafter
si son chemin est fourni. Un `PASS` signifie seulement « l'image expose les
briques » ; il n'autorise pas encore la production.

## Protocole de promotion

Chaque étape doit produire un artefact et laisser les deux rangs ainsi que
l'API sains :

1. construire une image SGLang SM121 depuis des commits complets et immuables,
   sans installer une tête de pull request au démarrage ; conserver digest,
   sources et sommes des patches dans le manifeste ;
2. épingler la révision complète du drafter
   `7d74cdd881ed7e32c31175984a67823127b66cfe` sur les deux nœuds et régler 7
   tokens draftés, jamais 8 ;
3. démarrer à 32K/C1, chauffer, comparer DFlash2 à NEXTN et sans spéculation sur
   les mêmes prompts et paramètres ;
4. à température 0, comparer les hash de sortie. Avec sampling, comparer la
   qualité/distribution sur un jeu suffisamment large plutôt que les textes
   octet pour octet ;
5. relever TTFT, tok/s par flux, débit agrégé et acceptation à C1/C2/C4/C6 ;
6. passer un froid 128K avec aiguilles, puis une vague de prefills concurrents,
   tout en vérifiant la marge mémoire minimale sur les deux nœuds ;
7. seulement après stabilité 128K, promouvoir `128k-dflash2` hors du niveau
   `experimental`. Les essais `256k-dflash2-eager` restent séparés : DFlash2
   accélère le décode, mais augmente la pression mémoire et ne résout pas le
   crash de contexte long.

Seuils de promotion proposés : zéro crash/redémarrage/retract anormal, smoke
chat + tools identique, acceptation chaude médiane ≥ 50 % sur code et ≥ 35 %
sur le mix, gain C1 ≥ 25 % contre MTP, p99 TTFT non régressé de plus de 10 % à
C4, et au moins 8 GiB de marge `MemAvailable` par nœud pendant le pire prefill.
Ces seuils sont ceux du cookbook, pas des garanties publiées par Inco AI.

## 512K : réponse nette

Les profils 512K actuels ne sont pas fiables : aucune requête froide 480/512K
n'a été validée sur cette recette TP2/GB10, et l'implémentation DFlash2 citée
sert 262 144 tokens sur deux Sparks. Son essai 1M utilise quatre Sparks. De
plus, [SGLang #36550](https://github.com/sgl-project/sglang/issues/36550)
confirme un crash de replay CUDA après de longs prefills ; le mode eager retire
ce défaut précis, pas la pression mémoire globale.

Ils restent donc en `PROFILE_TIER=quarantined`. `start-glm53.sh` les lance avec
un avertissement, et `bench-long-context.py` demande
`--allow-unsafe-profile` pour inscrire explicitement la prise de risque. Le
niveau ne sera promu qu'après une montée par paliers froids et des redémarrages
répétés sans erreur.
