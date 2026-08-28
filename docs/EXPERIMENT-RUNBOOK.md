# Runbook expérimental 2× DGX Spark

Ces profils démarrent tous directement. Exécutez les blocs dans l'ordre : le
profil arrêté au début d'un bloc est celui lancé dans le bloc précédent. Un
avertissement `experimental` ou `quarantined` décrit le niveau de preuve, pas
une interdiction.

Le client Python contourne lui-même les proxies pour `127.0.0.1`; aucun export
`no_proxy` n'est nécessaire. Si `API_KEY` est défini dans `.env.glm53`, exportez
la même valeur dans ce terminal.

## 1. Capacité 256K sans spéculation ni graphes

```bash
./stop-glm53.sh --profile 256k-mtp
./start-glm53.sh 256k
./status-glm53.sh 256k
for tokens in 180000 220000 240000; do
  ./bench-long-context.py --target-tokens "$tokens" --cold \
    --label "256k-base-${tokens}"
done
```

## 2. 256K avec graphes CUDA

```bash
./stop-glm53.sh --profile 256k
./start-glm53.sh 256k-graphs
./bench-glm53.py --warmup-runs 1 --runs 3 --concurrency 1
for tokens in 180000 220000 240000; do
  ./bench-long-context.py --allow-unsafe-profile --target-tokens "$tokens" \
    --cold --label "256k-graphs-${tokens}"
done
```

## 3. 256K MTP5 + graphes CUDA

```bash
./stop-glm53.sh --profile 256k-graphs
./start-glm53.sh 256k-mtp
./bench-glm53.py --warmup-runs 1 --runs 3 --concurrency 1
for tokens in 180000 220000 240000; do
  ./bench-long-context.py --allow-unsafe-profile --target-tokens "$tokens" \
    --cold --label "256k-mtp-${tokens}"
done
```

## 4. DFlash2 SGLang, FA4 puis concurrence

La préparation est à faire une fois. Elle construit l'image SGLang épinglée et
valide le SHA-256 du drafter sur les deux nœuds.

```bash
./stop-glm53.sh --profile 256k-mtp
./prepare-dflash2.sh 128k-dflash2
./start-glm53.sh 128k-dflash2
./bench-glm53.py --warmup-runs 1 --runs 3 --concurrency 1
./bench-glm53.py --warmup-runs 1 --runs 3 --concurrency 4
```

Balayage graphes CUDA jusqu'à batch 8 :

```bash
./stop-glm53.sh --profile 128k-dflash2
./start-glm53.sh 128k-dflash2-c8
for c in 1 2 4 5 6; do
  ./bench-glm53.py --warmup-runs 1 --runs 3 --concurrency "$c"
done
```

Si FA4 échoue au chargement sur SM121 :

```bash
./collect-glm53-report.sh --profile 128k-dflash2-c8 --label fa4-boot
./stop-glm53.sh --profile 128k-dflash2-c8
./start-glm53.sh 128k-dflash2-flashinfer
./bench-glm53.py --warmup-runs 1 --runs 3 --concurrency 1
```

## 5. DFlash2 sous pression 240K

```bash
./stop-glm53.sh --profile 128k-dflash2-c8
./start-glm53.sh 256k-dflash2-eager
for tokens in 180000 220000 240000; do
  ./bench-long-context.py --allow-unsafe-profile --target-tokens "$tokens" \
    --cold --label "256k-dflash2-${tokens}"
done
```

## 6. 384K BF16 KV + MTP + CP2

```bash
./stop-glm53.sh --profile 256k-dflash2-eager
./start-glm53.sh 384k-quality
for tokens in 300000 340000 360000; do
  ./bench-long-context.py --allow-unsafe-profile --target-tokens "$tokens" \
    --cold --label "384k-quality-${tokens}"
done
```

## 7. 512K MTP eager

```bash
./stop-glm53.sh --profile 384k-quality
./start-glm53.sh 512k-mtp-eager
./bench-glm53.py --warmup-runs 1 --runs 3 --concurrency 1
for tokens in 300000 400000 480000; do
  ./bench-long-context.py --allow-unsafe-profile --target-tokens "$tokens" \
    --cold --label "512k-mtp-eager-${tokens}"
done
```

## 8. 512K MTP + CP2 + graphes CUDA

```bash
./stop-glm53.sh --profile 512k-mtp-eager
./start-glm53.sh 512k-mtp-cp
./bench-glm53.py --warmup-runs 1 --runs 3 --concurrency 1
for tokens in 300000 400000 480000; do
  ./bench-long-context.py --allow-unsafe-profile --target-tokens "$tokens" \
    --cold --label "512k-mtp-cp-${tokens}"
done
```

## Après le moindre crash

Ne redémarrez pas avant cette commande :

```bash
./collect-glm53-report.sh --profile NOM_DU_PROFIL --label description-courte
```

Envoyez l'archive `.tar.gz` créée dans `results/diagnostics/`. Elle contient
les 2 000 dernières lignes des deux rangs, l'état et l'image des conteneurs,
`nvidia-smi`, la mémoire hôte et les 500 derniers événements kernel. Elle ne
capture ni `API_KEY`, ni l'environnement complet des conteneurs.
