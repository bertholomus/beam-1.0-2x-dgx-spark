#!/usr/bin/env bash
# beam-tf-serve.sh: the one command for Beam-1.0-501B-A23B EXL3 on two DGX Spark (GB10) nodes with TensorFold. Tensor
# parallel 2 over the 200G link: rank 0, the scheduler and the OpenAI API on $HEAD_HOST ($HOST:$PORT, served name $SERVED,
# Bearer key), rank 1 on $WORKER_HOST, the DFlash2 drafter always on (beam-tf.env).
# Run it on any machine that reaches both nodes with ssh keys and holds a clone of the TensorFold fork ($REPO); it keeps its
# files in ~/$TF_DIR/ on each node:
#   tf/beam-tf-serve.sh up         copy $REPO at $REV (a commit, never a working copy) and tf/ to both nodes, stop a running
#                                  lane, preflight, build the CUDA extensions (a throwaway container a node), start rank 1
#                                  then rank 0, each with its memory guard, wait for /v1/models
#   tf/beam-tf-serve.sh down       stop both ranks (each guard exits with its container; logs kept with a time stamp)
#   tf/beam-tf-serve.sh restart    down, then up
#   tf/beam-tf-serve.sh prebuild   copy, preflight and build only (no lane may run): a first build takes some minutes
#   tf/beam-tf-serve.sh status     containers and tree, KV pool, MemAvailable now and lowest since up, /v1/models
#   tf/beam-tf-serve.sh logs [N]   last N lines of rank 0 (default 40)
# Settings: tf/beam-tf.env. Any of them can be set for one run, e.g. PORT=8101 tf/beam-tf-serve.sh up.
set -uo pipefail
G=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$G/beam-tf.env"
ENVS="TF_TP_WORLD=2 NCCL_MAX_NCHANNELS=$NCCL_MAX_NCHANNELS NCCL_SOCKET_IFNAME=$NET_IF NCCL_IB_HCA=$NCCL_IB_HCA NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX NCCL_DEBUG=$NCCL_DEBUG GLIBC_TUNABLES=$GLIBC_TUNABLES $KNOBS"
T="\$HOME/$TF_DIR"
say() { echo "[beam-tf-serve $(date '+%H:%M:%S')] $*"; }
on() { local h=$1; shift; ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" "$@"; }

stage() {   # the commit's tree, the guard and the build script in ~/$TF_DIR/serve.new on both nodes (a running lane keeps serve/)
  local rev h; rev=$(git -C "$REPO" rev-parse --short=8 "$REV^{commit}") || { say "FAIL: no $REV in $REPO"; return 1; }
  for h in $WORKER_HOST $HEAD_HOST; do
    git -C "$REPO" archive --format=tar --prefix=TensorFold/ "$rev" |
      on "$h" "rm -rf $T/serve.new && mkdir -p $T/serve.new && tar xf - -C $T/serve.new" || { say "FAIL: copy to $h"; return 1; }
    on "$h" "cat > $T/serve.new/memguard.sh && chmod +x $T/serve.new/memguard.sh && echo $rev > $T/serve.new/REV" \
      < "$G/memguard.sh" || { say "FAIL: copy to $h"; return 1; }
    on "$h" "cat > $T/serve.new/build.sh" < "$G/beam-tf-build.sh" || { say "FAIL: copy to $h"; return 1; }
  done
  say "staged $REV = $rev on $WORKER_HOST and $HEAD_HOST"
}
swap() { local h; for h in $WORKER_HOST $HEAD_HOST; do
  on "$h" "rm -rf $T/serve && mv $T/serve.new $T/serve" || { say "FAIL: swap on $h"; return 1; }; done; }

preflight() {   # both GPUs free, enough MemAvailable, the weights in place, the same image on both, the port free, the key
  local h out id ids="" bad=0
  for h in $WORKER_HOST $HEAD_HOST; do
    out=$(on "$h" "n=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); [ \$n = 0 ] || echo \"GPU busy: \$n compute process(es)\"
      a=\$(awk '/^MemAvailable:/{print int(\$2/1048576)}' /proc/meminfo); [ \$a -ge $MIN_START_GIB ] || echo \"MemAvailable \$a GiB < $MIN_START_GIB GiB\"
      [ -f $PACK/config.json ] || echo 'no pack at $PACK'; [ -f $DRAFT/config.json ] || echo 'no drafter at $DRAFT'
      [ $h != $HEAD_HOST ] || ! ss -ltnH 2>/dev/null | awk '{print \$4}' | grep -qE ':$PORT\$' || echo 'port $PORT in use'
      [ $h != $HEAD_HOST ] || [ -s \$HOME/$API_KEY_FILE ] || (mkdir -p \"\$(dirname \$HOME/$API_KEY_FILE)\" && umask 077 &&
        head -c 24 /dev/urandom | base64 | tr -d '/+=' > \$HOME/$API_KEY_FILE) || echo 'no API key file'") || { say "FAIL: $h unreachable"; return 1; }
    [ -n "$out" ] && { echo "$out" | sed "s/^/  $h: /"; bad=1; }
    id=$(on "$h" "docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null"); [ -n "$id" ] || { say "FAIL: no $IMAGE on $h"; bad=1; }
    ids+="$id "
  done
  [ "$(echo $ids | tr ' ' '\n' | sort -u | wc -l)" = 1 ] || { say "FAIL: image differs between nodes"; bad=1; }
  [ $bad = 0 ] && say "preflight ok: image ${ids:0:19}, both nodes free" || { say "FAIL: preflight"; return 1; }
}

prebuild() {   # the CUDA extensions built (or found current) on both nodes before any weight loads, each in a throwaway
               # container with its guard: nvcc inside a starting rank would run beside the loaded weights
  local h bad=0
  for h in $WORKER_HOST $HEAD_HOST; do
    on "$h" "docker rm -f beam-tf-serve-build >/dev/null 2>&1; mkdir -p $T/logs $T/cache; : > $T/logs/mem-serve-build.log
      docker run -d --name beam-tf-serve-build --gpus all -v $T/serve:/tfw -v $T/logs:/w/logs \
        -v $T/cache:/root/.cache -e TORCH_EXTENSIONS_DIR=$EXT_DIR -e PYTHONDONTWRITEBYTECODE=1 -w /tfw $IMAGE \
        bash /tfw/build.sh >/dev/null &&
      (nohup $T/serve/memguard.sh beam-tf-serve-build $GUARD_MIB $T/logs/mem-serve-build.log >/dev/null 2>&1 &)" ||
      { say "FAIL: no build container on $h"; bad=1; }
  done
  for h in $WORKER_HOST $HEAD_HOST; do
    on "$h" "docker wait beam-tf-serve-build >/dev/null 2>&1; docker logs beam-tf-serve-build > $T/logs/beam-tf-serve-build.log 2>&1
      docker rm -f beam-tf-serve-build >/dev/null 2>&1; grep -q BUILD_OK $T/logs/beam-tf-serve-build.log" ||
      { say "FAIL: CUDA extensions on $h"; on "$h" "tail -n 4 $T/logs/beam-tf-serve-build.log $T/logs/mem-serve-build.log" |
        cut -c1-300 | sed "s/^/  $h: /"; bad=1; }
  done
  [ $bad = 0 ] || return 1
  say "CUDA extensions current on both nodes ($EXT_DIR)"
}

start() {   # start HOST RANK: the rank's container (rank 0 with the key file mounted read-only), then its memory guard
  local h=$1 r=$2 e envs="" key="" auth=""
  for e in $ENVS; do envs+="-e $e "; done
  [ "$r" = 0 ] && { key="-v \$HOME/$API_KEY_FILE:/run/beam-api-key:ro"; auth="--api-key-file /run/beam-api-key"; }
  on "$h" "docker rm -f beam-tf-serve-r$r >/dev/null 2>&1; mkdir -p $T/logs $T/cache; : > $T/logs/mem-serve-r$r.log
    docker run -d --name beam-tf-serve-r$r --rm --gpus all --network host --ipc host --ulimit memlock=-1 --ulimit stack=67108864 \
      --cap-add IPC_LOCK \$( [ -d /dev/infiniband ] && echo --device /dev/infiniband ) -v $T/serve:/tfw \
      -v $T/logs:/w/logs -v $PACK:/models/pack:ro -v $DRAFT:/models/draft:ro -v $T/cache:/root/.cache $key \
      -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e TRITON_CACHE_DIR=/root/.cache/triton \
      -e CUDA_CACHE_PATH=/root/.cache/nv-ComputeCache -e CUDA_CACHE_MAXSIZE=4294967296 -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
      -e TORCH_EXTENSIONS_DIR=$EXT_DIR \
      $envs -w /tfw $IMAGE bash -c 'pip install --no-deps --no-build-isolation -q -e /tfw/TensorFold >/w/logs/beam-tf-serve-r$r.pip 2>&1; \
      exec tensorfold serve /models/pack --backend cuda --tp 2 --master $HEAD_IP --master-port $MASTER_PORT --host $HOST --port $PORT \
      --name $SERVED --context $CONTEXT --drafter /models/draft --rank $r $auth > /w/logs/beam-tf-serve-r$r.log 2>&1' >/dev/null &&
    (nohup $T/serve/memguard.sh beam-tf-serve-r$r $GUARD_MIB $T/logs/mem-serve-r$r.log >/dev/null 2>&1 &)"
}

down() {
  local h r
  for h in $HEAD_HOST:0 $WORKER_HOST:1; do
    r=${h##*:}; h=${h%%:*}
    on "$h" "f=$T/logs/beam-tf-serve-r$r.log; docker inspect beam-tf-serve-r$r >/dev/null 2>&1 && [ -f \$f ] &&
      cp \$f $T/logs/beam-tf-serve-r$r-\$(date -u +%Y%m%dT%H%MZ).log; docker rm -f beam-tf-serve-r$r >/dev/null 2>&1; true"
  done
  say "down: beam-tf-serve-r0 ($HEAD_HOST) and beam-tf-serve-r1 ($WORKER_HOST) stopped"
}

running() { local h n=0; for h in $HEAD_HOST:0 $WORKER_HOST:1; do
  on "${h%%:*}" "docker inspect beam-tf-serve-r${h##*:} >/dev/null 2>&1" && n=$((n + 1)); done; echo $n; }

models() { on "$HEAD_HOST" "curl -s -m 5 -o /dev/null -w '%{http_code}' -H \"Authorization: Bearer \$(cat \$HOME/$API_KEY_FILE)\" http://$HOST:$PORT/v1/models"; }

status() {
  local h r
  for h in $HEAD_HOST:0 $WORKER_HOST:1; do
    r=${h##*:}; h=${h%%:*}
    on "$h" "s=\$(docker inspect -f '{{.State.Status}} since {{.State.StartedAt}}' beam-tf-serve-r$r 2>/dev/null)
      echo \"\$(hostname) rank $r: \${s:-absent}, tree \$(cat $T/serve/REV 2>/dev/null || echo none)\"
      grep -E 'KV pool [0-9]+ positions' $T/logs/beam-tf-serve-r$r.log 2>/dev/null | tail -1 | cut -c1-200 | sed 's/^/  /'
      echo \"  MemAvailable now \$(awk '/^MemAvailable:/{print int(\$2/1024)}' /proc/meminfo) MiB, lowest since up \$(awk 'NF==2 && \$2+0>0 {if(m==\"\"||\$2+0<m)m=\$2+0} END {print m}' $T/logs/mem-serve-r$r.log 2>/dev/null) MiB\""
  done
  echo "  /v1/models on $HEAD_HOST $HOST:$PORT: $(models)"
}

up() {
  stage || return 1
  down >/dev/null; sleep 3
  swap && preflight && prebuild || return 1
  start "$WORKER_HOST" 1 || { say "FAIL: rank 1 did not start"; down; return 1; }
  start "$HEAD_HOST" 0 || { say "FAIL: rank 0 did not start"; down; return 1; }
  say "up: $REV on $HEAD_HOST (rank 0, $HOST:$PORT) + $WORKER_HOST (rank 1); $KNOBS; guard $GUARD_MIB MiB"
  local t0=$SECONDS s0 s1 h
  while (( SECONDS - t0 < READY_MIN * 60 )); do
    sleep 10
    [ "$(models)" = 200 ] && { say "UP_OK after $((SECONDS - t0)) s"; status; return 0; }
    s0=$(on "$HEAD_HOST" 'docker inspect -f {{.State.Status}} beam-tf-serve-r0 2>/dev/null'); s1=$(on "$WORKER_HOST" 'docker inspect -f {{.State.Status}} beam-tf-serve-r1 2>/dev/null')
    if [ "$s0 $s1" != "running running" ]; then
      say "UP_FAIL: $HEAD_HOST [$s0] $WORKER_HOST [$s1]"
      for h in $HEAD_HOST:0 $WORKER_HOST:1; do on "${h%%:*}" "grep -h -E 'Error|Traceback|GUARD' $T/logs/beam-tf-serve-r${h##*:}.log \
        $T/logs/beam-tf-serve-r${h##*:}.pip $T/logs/mem-serve-r${h##*:}.log 2>/dev/null | head -6 | cut -c1-300 | sed 's/^/  ${h%%:*}: /'"; done
      down; return 1
    fi
  done
  say "UP_FAIL: not ready after $READY_MIN min"; down; return 1
}

case ${1:-} in
  up) up ;; down) down ;; restart) down; up ;; status) status ;;
  prebuild) [ "$(running)" = 0 ] || { say "FAIL: a lane runs; prebuild only with both ranks down"; exit 1; }
            stage && swap && preflight && prebuild ;;
  logs) on "$HEAD_HOST" "tail -n ${2:-40} $T/logs/beam-tf-serve-r0.log" ;;
  *) sed -n 2,15p "$0"; exit 2 ;;
esac
