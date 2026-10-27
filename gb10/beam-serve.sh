#!/usr/bin/env bash
# beam-serve.sh: the one command for the Beam-1.0-501B-A23B EXL3 lane on two DGX Spark (GB10) nodes.
# vLLM TP2 over the 200G fabric: rank 0 + OpenAI API on $HEAD_HOST, rank 1 headless on $WORKER_HOST (beam-gb10.env). When the API
# runs on the other node's fabric address, $API_FWD_HOST relays its 127.0.0.1:$PORT to it (gb10/apifwd.py, user unit beam-fwd).
# Run it on any machine that reaches both nodes with ssh keys; it keeps its files in ~/beam-serve/ on each node:
#   gb10/beam-serve.sh up         copy gb10/ to both nodes, preflight, start rank 1 then rank 0, memory traces, relay, wait for /v1/models
#   gb10/beam-serve.sh down       stop rank 0, rank 1, the relay and the memory traces
#   gb10/beam-serve.sh restart    down, then up
#   gb10/beam-serve.sh status     containers, /v1/models, KV pool, MemAvailable now and lowest since up
#   gb10/beam-serve.sh logs [N]   last N lines of rank 0 (default 40)
# Settings: gb10/beam-gb10.env. Any of them can be set for one run, e.g. EAGER=0 SPEC=dflash gb10/beam-serve.sh up.
# EXL3_*, THOR_*, VLLM_*, NCCL_*, BEAM_GB10_*, MALLOC_*, GLIBC_TUNABLES and PYTORCH_CUDA_ALLOC_CONF from the caller's environment go to both nodes too, so both ranks always run
# the same settings. Guards: refuses to start when a node is unreachable, the image differs between nodes, the GPU is busy, or
# MemAvailable is under MIN_START_GIB; the memory trace kills the container below GUARD_MIB (memtrace.sh).
set -uo pipefail
G=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$G/beam-gb10.env"
MIN_START_GIB=${MIN_START_GIB:-110}   # free memory each node needs before a start (weights ~90 GiB + KV pool + activations)
GUARD_MIB=${GUARD_MIB:-4096}          # memtrace.sh kills the container when MemAvailable stays under this for 3 s (wedge guard)
READY_MIN=${READY_MIN:-60}
VARS="IMAGE NAME PACK DRAFT RECIPE CACHE_DIR API_KEY_FILE HEAD_HOST HEAD_IP WORKER_HOST WORKER_IP MASTER_PORT NET_IF NCCL_IB_HCA
NCCL_DEBUG NCCL_MAX_NCHANNELS GLIBC_TUNABLES API_HOST API_FWD_HOST PORT SERVED MAX_MODEL_LEN MAX_NUM_SEQS MAX_BATCHED UTIL KV_MEM KV_DTYPE KV_OFFLOAD SPEC SPEC_K
SPEC_ATTN CAPTURE_SIZES EAGER NO_CUSTOM_AR HOSTPLAN XMOE_EXT THOR_EMBED_HOST EXTRA_ARGS"
SYNC="beam-gb10.env serve-node.sh memtrace.sh meminfo-trace.sh beam_gb10_hooks.py beam_gb10.pth apifwd.py probe5.py rows.py maxctx.py final-run.sh
report.sh kl-run.sh get-kld-base.sh hfrange.py gated_rows.py gated-run.sh"

exports() {   # the effective settings, as export lines for the remote shell
  local v k val
  for v in $VARS; do [[ -n ${!v+x} ]] && printf 'export %s=%q\n' "$v" "${!v}"; done
  env | grep -E '^(EXL3_|THOR_|VLLM_|NCCL_|BEAM_GB10_|MALLOC_|GLIBC_TUNABLES|PYTORCH_CUDA_ALLOC_CONF)[A-Z0-9_]*=' | while IFS='=' read -r k val; do printf 'export %s=%q\n' "$k" "$val"; done
}
remote() { local h=$1; shift; { exports; printf '%s\n' "$@"; } | ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" bash -s; }
say() { echo "[beam-serve $(date '+%H:%M:%S')] $*"; }

sync_gb10() {
  local h f; for h in $WORKER_HOST $HEAD_HOST; do
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" mkdir -p beam-serve/gb10 beam-serve/logs || { say "FAIL: $h unreachable"; return 1; }
    for f in $SYNC; do [[ -f $G/$f ]] && scp -q "$G/$f" "$h:beam-serve/gb10/$f"; done
  done
}

preflight() {
  local h id ids="" bad=0
  for h in $WORKER_HOST $HEAD_HOST; do
    id=$(remote "$h" 'docker image inspect -f "{{.Id}}" "$IMAGE" 2>/dev/null; true') || { say "FAIL: $h unreachable"; return 1; }
    [[ -n $id ]] || { say "FAIL: image $IMAGE missing on $h"; bad=1; }
    ids+="$id "
    remote "$h" '[[ -f $PACK/config.json ]] || echo "no $PACK/config.json"
      [[ $SPEC != dflash || -f $DRAFT/config.json ]] || echo "no drafter at $DRAFT"
      for m in ${XMOE_EXT:-} xmoe_do8 x8rh f8rh; do [[ -d $RECIPE/kernels/build/$m ]] || echo "no kernels/build/$m (gb10/build-kernels.sh)"; done
      n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); [[ $n == 0 ]] || echo "GPU busy: $n compute process(es)"
      a=$(awk "/^MemAvailable:/{print int(\$2/1048576)}" /proc/meminfo); [[ $a -ge '"$MIN_START_GIB"' ]] || echo "MemAvailable ${a} GiB < '"$MIN_START_GIB"' GiB"' \
      | sed "s/^/  $h: /" | grep . && bad=1
  done
  [[ $(echo $ids | tr ' ' '\n' | sort -u | wc -l) == 1 ]] || { say "FAIL: image differs between nodes: $ids"; bad=1; }
  [[ $bad == 0 ]] && say "preflight ok: image ${ids%% *}, both nodes free" || { say "FAIL: preflight"; return 1; }
}

down() {   # keeps each container's log as beam-serve/logs/container-<tag>-<host>.log before removing it
  local h; for h in $HEAD_HOST $WORKER_HOST; do
    remote "$h" 'tag=$(cat beam-serve/logs/mem-current-tag 2>/dev/null || echo manual)
      docker inspect "$NAME" >/dev/null 2>&1 && docker logs --timestamps "$NAME" > beam-serve/logs/container-$tag-$(hostname).log 2>&1
      docker rm -f "$NAME" >/dev/null 2>&1; systemctl --user stop beam-mem beam-fwd >/dev/null 2>&1; true'
  done; say "down: $NAME stopped on $HEAD_HOST and $WORKER_HOST (logs in beam-serve/logs/container-<tag>-<host>.log)"
}

up() {
  local tag; tag=${TAG:-$(date -u +%Y%m%dT%H%MZ)}
  say "up: EAGER=$EAGER SPEC=$SPEC SPEC_K=$SPEC_K NCCL_MAX_NCHANNELS=$NCCL_MAX_NCHANNELS KV_MEM=$KV_MEM UTIL=$UTIL MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS MAX_BATCHED=$MAX_BATCHED${EXTRA_ARGS:+ EXTRA_ARGS=$EXTRA_ARGS} guard ${GUARD_MIB} MiB tag $tag"
  sync_gb10 || return 1
  down >/dev/null; sleep 3
  preflight || return 1
  local h; for h in $WORKER_HOST $HEAD_HOST; do
    remote "$h" "echo $tag > beam-serve/logs/mem-current-tag; systemctl --user stop beam-mem >/dev/null 2>&1; systemctl --user reset-failed beam-mem >/dev/null 2>&1
      systemd-run --user --unit beam-mem --setenv=GUARD_MIB=$GUARD_MIB bash \$HOME/beam-serve/gb10/memtrace.sh $tag >/dev/null 2>&1 || echo 'memory trace did not start'"
  done
  remote "$WORKER_HOST" "cd beam-serve && NODE_RANK=1 bash gb10/serve-node.sh > logs/serve-node-$tag.out 2>&1" || { say "FAIL: rank 1 did not start"; return 1; }
  remote "$HEAD_HOST" "cd beam-serve && NODE_RANK=0 bash gb10/serve-node.sh > logs/serve-node-$tag.out 2>&1" || { say "FAIL: rank 0 did not start"; return 1; }
  if [[ -n ${API_FWD_HOST:-} && $API_FWD_HOST != "$HEAD_HOST" ]]; then
    remote "$API_FWD_HOST" "systemctl --user reset-failed beam-fwd >/dev/null 2>&1
      systemd-run --user --unit beam-fwd --setenv=FWD_TARGET=$API_HOST --setenv=FWD_PORT=$PORT python3 \$HOME/beam-serve/gb10/apifwd.py >/dev/null 2>&1 || echo 'relay did not start'"
  fi
  local t0=$SECONDS s4 s3
  while (( SECONDS - t0 < READY_MIN * 60 )); do
    s4=$(remote "$HEAD_HOST" 'echo $(docker inspect -f "{{.State.Status}}" "$NAME" 2>/dev/null) $(curl -s -m 3 -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $(cat "$API_KEY_FILE")" http://$API_HOST:$PORT/v1/models)')
    s3=$(remote "$WORKER_HOST" 'docker inspect -f "{{.State.Status}}" "$NAME" 2>/dev/null')
    if [[ $s4 == *" 200" ]]; then say "UP_OK after $((SECONDS - t0)) s"; status; return 0; fi
    if [[ "$s4 $s3" != running*running* ]]; then say "UP_FAIL: $HEAD_HOST [$s4] $WORKER_HOST [$s3]"; fail_logs; down; return 1; fi
    sleep 10
  done
  say "UP_FAIL: not ready after $READY_MIN min"; fail_logs; down; return 1
}
fail_logs() { local h; for h in $HEAD_HOST $WORKER_HOST; do echo "== $h"; remote "$h" 'docker logs "$NAME" 2>&1 | tr "\r" "\n" |
  grep -E "Error|ERROR|Exception|Traceback|out of memory|GUARD" | grep -v "NCCL INFO" | head -n 8 | cut -c1-300'; done; }

status() {
  remote "$HEAD_HOST" 'echo "$(hostname) rank 0: $(docker inspect -f "{{.State.Status}} since {{.State.StartedAt}}" "$NAME" 2>/dev/null || echo absent)"
    echo "  /v1/models: $(curl -s -m 3 -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $(cat "$API_KEY_FILE" 2>/dev/null)" http://$API_HOST:$PORT/v1/models)"
    docker logs "$NAME" 2>&1 | grep -E "GPU KV cache size|Maximum concurrency|Model loading took|Loading weights took" | tail -n 4 | sed "s/^/  /"'
  remote "$WORKER_HOST" 'echo "$(hostname) rank 1: $(docker inspect -f "{{.State.Status}} since {{.State.StartedAt}}" "$NAME" 2>/dev/null || echo absent)"'
  [[ -n ${API_FWD_HOST:-} && $API_FWD_HOST != "$HEAD_HOST" ]] && remote "$API_FWD_HOST" 'echo "$(hostname) relay 127.0.0.1:$PORT -> $API_HOST:$PORT ($(systemctl --user is-active beam-fwd)): /v1/models $(curl -s -m 3 -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $(cat "$API_KEY_FILE" 2>/dev/null)" http://127.0.0.1:$PORT/v1/models)"'
  local h; for h in $HEAD_HOST $WORKER_HOST; do
    remote "$h" 'tag=$(cat beam-serve/logs/mem-current-tag 2>/dev/null); f=beam-serve/logs/mem-$(hostname)-$tag.log
      now=$(awk "/^MemAvailable:/{print int(\$2/1024)}" /proc/meminfo); low=$( [[ -f $f ]] && sort -k2 -n "$f" | head -1 | cut -d" " -f2)
      echo "$(hostname) MemAvailable now ${now} MiB, lowest since up ${low:-?} MiB (trace $f, $(systemctl --user is-active beam-mem))"'
  done
}

logs() { remote "$HEAD_HOST" "docker logs --tail ${1:-40} \"\$NAME\" 2>&1 | tr '\r' '\n' | grep -v 'NCCL INFO' | tail -n ${1:-40}"; }

case ${1:-} in
  up) up ;;
  down) down ;;
  restart) down; up ;;
  status) status ;;
  logs) logs "${2:-40}" ;;
  *) sed -n 2,13p "$0"; exit 2 ;;
esac
