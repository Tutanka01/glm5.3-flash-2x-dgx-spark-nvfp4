# Vérification du fabric RoCE

La recipe suppose que le lien TP=2 fonctionne déjà. Les noms d'interface, HCA et l'index GID sont spécifiques à chaque nœud et peuvent changer après une mise à jour firmware.

Sur chaque machine :

```bash
ip -br link
ip -br -4 addr
ibdev2netdev
ibstat
```

Pour afficher les GID et repérer le RoCE v2 correspondant à l'IPv4 du lien :

```bash
for dev in /sys/class/infiniband/*; do
  for port in "$dev"/ports/*; do
    printf '\n%s port %s\n' "$(basename "$dev")" "$(basename "$port")"
    for gid in "$port"/gids/*; do
      idx="$(basename "$gid")"
      printf '%s  %-39s  %s\n' \
        "$idx" \
        "$(cat "$gid")" \
        "$(cat "$port/gid_attrs/types/$idx" 2>/dev/null || true)"
    done
  done
done
```

Les variables importantes sont :

```text
VLLM_HOST_IP / WORKER_VLLM_HOST_IP
NCCL_IB_HCA / WORKER_NCCL_IB_HCA
NCCL_SOCKET_IFNAME / WORKER_NCCL_SOCKET_IFNAME
TP_SOCKET_IFNAME / WORKER_TP_SOCKET_IFNAME
GLOO_SOCKET_IFNAME / WORKER_GLOO_SOCKET_IFNAME
NCCL_IB_GID_INDEX / WORKER_NCCL_IB_GID_INDEX
```

`GLOO_SOCKET_IFNAME` doit pointer vers le fabric et non loopback. Une erreur `Gloo connectFullMesh Connection refused` est généralement un mauvais bind d'interface ou une IP de rang incorrecte.

Teste enfin les routes dans les deux sens :

```bash
# head
ip route get <WORKER_VLLM_HOST_IP>

# worker
ip route get <VLLM_HOST_IP>
```

`./doctor-glm53.sh` automatise ces contrôles, inspecte les GID sysfs et échoue si l'IP n'est pas affectée à l'interface Gloo sélectionnée.

