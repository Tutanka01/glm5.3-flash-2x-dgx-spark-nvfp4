# Vérification du fabric RoCE

La recette exige un chemin réseau cohérent entre les deux rangs : l'IP configurée, l'interface Gloo/NCCL, le HCA RDMA et la route vers le pair doivent tous désigner le même lien.

## Variables

```text
HEAD_FABRIC_IP / WORKER_FABRIC_IP
NCCL_IB_HCA / WORKER_NCCL_IB_HCA
NCCL_SOCKET_IFNAME / WORKER_NCCL_SOCKET_IFNAME
TP_SOCKET_IFNAME / WORKER_TP_SOCKET_IFNAME
GLOO_SOCKET_IFNAME / WORKER_GLOO_SOCKET_IFNAME
NCCL_IB_ADDR_RANGE / WORKER_NCCL_IB_ADDR_RANGE
```

Reprenez de préférence ces valeurs d'une recette TP=2 déjà validée sur les mêmes machines.

## Interfaces et routes

Sur chaque nœud :

```bash
ip -br link
ip -br -4 addr
ibdev2netdev
ibstat
```

Puis contrôlez le chemin vers le pair :

```bash
# head
ip -4 route get <WORKER_FABRIC_IP>

# worker
ip -4 route get <HEAD_FABRIC_IP>
```

La sortie doit contenir exactement l'interface configurée et l'IP locale attendue. Exemple :

```text
192.168.100.11 dev enp1s0f0np0 src 192.168.100.10
```

Si la commande renvoie une autre interface ou une autre source — par exemple une IP `.14` alors que `.10` est configurée — corrigez `.env.glm53` avant le démarrage. Deux interfaces placées dans le même sous-réseau peuvent créer ce type de route ambiguë.

## GID RoCE

Le runtime contient NCCL ≥ 2.21. NVIDIA recommande alors de ne pas définir `NCCL_IB_GID_INDEX` : NCCL sélectionne dynamiquement un GID RoCE v2 selon le HCA, la famille d'adresse et la plage configurée.

Conservez :

```bash
NCCL_IB_ROCE_VERSION_NUM=2
NCCL_IB_ADDR_FAMILY=AF_INET
NCCL_IB_ADDR_RANGE=192.168.100.0/24
```

Pour afficher les GID disponibles sans provoquer d'erreur sur les entrées vides :

```bash
for dev in /sys/class/infiniband/*; do
  for port in "$dev"/ports/*; do
    printf '\n%s port %s\n' "$(basename "$dev")" "$(basename "$port")"
    for gid in "$port"/gids/*; do
      idx="$(basename "$gid")"
      value="$(cat "$gid" 2>/dev/null || true)"
      type="$(cat "$port/gid_attrs/types/$idx" 2>/dev/null || true)"
      ndev="$(cat "$port/gid_attrs/ndevs/$idx" 2>/dev/null || true)"
      case "$value" in ""|"::") continue ;; esac
      printf '%s  %-39s  %-12s  %s\n' "$idx" "$value" "$type" "$ndev"
    done
  done
done
```

`./doctor-glm53.sh` automatise ces contrôles. Il échoue si :

- l'IP n'est pas affectée à l'interface Gloo ;
- la route vers le pair utilise une autre interface ou une autre source ;
- le HCA configuré n'existe pas ;
- aucun GID RoCE v2 rempli n'est associé à l'interface sélectionnée.

Référence : [guide de dépannage réseau NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/networking_troubleshooting.html).
