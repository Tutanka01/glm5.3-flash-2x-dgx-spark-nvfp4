# Mesure de la consommation électrique — `bench-power.py`

Aucune mesure de puissance n'existait dans le journal des benchmarks : ce
script comble ce manque. Il échantillonne la puissance des GPU au fil de
l'eau autour d'une commande (bench, long contexte, prompts), intègre
l'énergie consommée et sépare proprement l'idle de la charge utile. Il ne
mesure **pas** la prise murale : voir « Limites » plus bas.

Le script est stdlib-only (`python3` suffit, `pynvml` optionnel), et doit
être lancé **sur les nœuds GB10**, pas sur un Mac — sans `nvidia-smi` ni
driver NVML il refuse de démarrer avec un message explicite.

## Démarrage rapide

Sur le head (ou le worker), depus la racine du dépôt :

```bash
# Que détecte-t-on ?
./bench-power.py --list-gpus

# Relevé idle : combien consomment les GPU au repos (modèle chargé) ?
./bench-power.py --watch --label idle --duration 60

# Autour d'un bench : baseline idle 10 s → bench → idle 10 s
./bench-power.py --label c6 \
  -- python3 bench-glm53.py --runs 3 --concurrency 6

# Long contexte froid avec baseline plus longue
./bench-power.py --label long-200k --idle-window 15 \
  -- python3 bench-long-context.py --target-tokens 200000 --cold --label 256k-safe
```

Chaque exécution écrit trois artefacts dans `results/` (ignoré par Git) :

| Artefact | Contenu |
|---|---|
| `glm53-power-<label>-<stamp>.json` | Résumé : stats par GPU (moyenne, pic, p95), énergie (J/Wh) et méthode, phases idle/charge, baseline idle, excès de charge |
| `glm53-power-<label>-<stamp>.jsonl` | Flux brut ligne par ligne : `sample` (t, phase, watts, util, temp, horloges, compteur mJ), `marker`, `event` — `tail -f` + `jq` pour suivre en direct |
| `glm53-power-<label>-<stamp>.svg` | Courbe de la puissance totale (bandes grises = phases idle, pointillés = markers) |

## Séparer les choses

- **Baseline idle vs charge** : par défaut (`--idle-window 10`), le script
  échantillonne 10 s avant et 10 s après la commande enveloppée. Le résumé
  rapporte la puissance idle moyenne, la puissance moyenne sous charge et
  l'**excès de charge** (W et Wh) — c'est l'énergie réellement imputable au
  travail, pas au modèle qui dort. `--idle-window 0` désactive.
- **GPU vs système** : la puissance GPU vient de NVML (pynvml en priorité,
  sinon polling `nvidia-smi` ; le champ `power.draw.average` est utilisé
  quand le driver le propose). Si le kernel expose des compteurs RAPL
  (`/sys/class/powercap`), une section `system` séparée rapporte l'énergie
  package CPU. Sur GB10, RAPL est généralement absent : la section est alors
  simplement absente du résumé.
- **Échantillonnage vs analyse** : le `.jsonl` est la vérité terrain
  horodatée ; le `.json` n'est qu'une agrégation. Toute analyse a posteriori
  (corrélation avec les artefacts de bench par timestamp, découpage par
  phase personnalisé) repart du `.jsonl`.
- **Énergie exacte vs intégrée** : si le driver expose le compteur d'énergie
  NVML (mJ), l'énergie provient du compteur (méthode `counter`) et la
  trapèze est conservée en contre-vérification ; sinon, intégration
  trapézoïdale des échantillons (méthode `trapezoid`), avec exclusion des
  trous d'échantillonnage > 5 s (comptabilisés dans `excluded_gap_seconds`).

## Attribuer l'énergie prompt par prompt

En mode `--watch`, passez `--markers FICHIER` : chaque ligne ajoutée au
fichier devient un `marker` horodaté dans le flux, sans toucher au sampler.

```bash
: > /tmp/power-marks.txt
./bench-power.py --watch --label prompts --markers /tmp/power-marks.txt &
# ... puis, à côté :
echo "prompt 1 : coding C6" >> /tmp/power-marks.txt
./bench-glm53.py --runs 1 --concurrency 6
echo "prompt 2 : long 200k" >> /tmp/power-marks.txt
python3 bench-long-context.py --target-tokens 200000 --cold
```

Le `.svg` trace les markers en pointillés ; le `.jsonl` permet de découper
l'énergie entre deux markers (intervalle × puissance moyenne).

## Deux nœuds = deux samplers

En TP=2 sur deux machines, chaque nœud ne voit que son propre GPU. Pour une
énergie de cluster, lancez un sampler **sur chaque nœud** (mêmes options,
labels distincts `head`/`worker`) et sommez les énergies du résumé.

## Options principales

| Option | Défaut | Rôle |
|---|---|---|
| `--interval` | 0.5 s | période d'échantillonnage (0.05–30 s ; ≥ 0.2 s conseillé si le backend retombe sur le polling `nvidia-smi`) |
| `--idle-window` | 10 s | baseline idle avant/après la commande enveloppée, 0 pour désactiver |
| `--timeout` | 0 | arrêt brutal de la commande enveloppée après N s |
| `--duration` | 0 | arrêt automatique du mode `--watch` après N s |
| `--label` | run | tag dans les noms de fichiers |
| `--gpu` | tous | indices GPU à échantillonner, ex. `--gpu 0,1` |
| `--markers` | — | fichier de markers (créé s'il manque) |
| `--no-jsonl` / `--no-chart` / `--quiet` | — | réduisent les sorties |

Le script gère Ctrl+C/SIGTERM proprement : la commande enveloppée reçoit
SIGTERM (puis SIGKILL après 5 s), le résumé et la courbe sont écrits
quand même, et le code de sortie de la commande est propagé (124 si
`--timeout`, 130/143 si signal).

## Journaliser dans BENCHMARKS.md

Une ligne type pour [BENCHMARKS.md](BENCHMARKS.md), dans la section de la
lane concernée :

```markdown
| 2026-08-31 | `128k-dflash2-c8` | bench C6 ×3 sous `bench-power.py --label c6` | — | — | — | — | GPU mean 238 W, peak 341 W, 6.8 Wh/bench (idle 39 W, excès 5.9 Wh) | `glm53-power-c6-20260831-*.json` |
```

## Limites honnêtes

- La puissance NVML est la puissance **carte GPU** — un seul GPU GB10 par
  nœud, TP=2 oblige ; CPU Grace, DRAM, NICs, disques et pertes
  d'alimentation ne sont pas couverts. L'énergie cluster réelle est donc
  supérieure aux chiffres ici — dans un facteur constant à première
  approximation, ce qui suffit pour **comparer** les profils.
- L'« idle » mesuré est le serveur SGLang/vLLM chargé et à vide : c'est le
  bon baseline d'exploitation, pas un GPU au repos absolu.
- Les compteurs RAPL, s'ils existent, couvrent le package CPU uniquement.
- `power.draw` instantané peut manquer sur certains drivers ; le script
  sonde `power.draw.average` au démarrage et retombe proprement.

## Tests locaux

`tests/test_bench_power.py` couvre l'intégration trapézoïdale, le compteur,
les markers, le parsing CSV `nvidia-smi`, et fait tourner les modes
wrap/watch avec un collecteur factice (aucun GPU requis) :

```bash
python3 -m unittest -v tests/test_bench_power.py
# ou la suite complète : ./tests/run-local.sh
```
