#!/usr/bin/env bash
# serve-node.sh: GB10 port of diffbot's recipe/serve/serve.sh. Starts ONE node's part of the vLLM TP2 lane for
# Beam-1.0-501B-A23B EXL3 in a detached container named $NAME:
#   NODE_RANK=0 on $HEAD_HOST: engine + OpenAI API on $API_HOST:$PORT (its 200G-link address) + TP rank 0
#   NODE_RANK=1 on $WORKER_HOST: headless worker, TP rank 1
# Start rank 1 first, then rank 0 (gb10/beam-serve.sh does both). Logs: docker logs $NAME.
# Changes vs upstream serve.sh (one host, 2x RTX PRO 6000): vLLM native multi-node (--nnodes 2 --node-rank, --headless on rank 1),
# /dev/infiniband + IPC_LOCK + memlock for RoCE, NCCL/Gloo pinned to the 200G fabric, API on a fabric/loopback address only, detached container (logs kept),
# GB10 defaults in beam-gb10.env (no host-RAM experts, THOR_EMBED_HOST=0, no CPU KV tier, NCCL all-reduce, UTIL for unified memory),
# the GB10 load hook (beam_gb10_hooks.py: safetensors tensors copied to RAM before the GPU copy) and vLLM usage stats off.
# Knob exports are upstream's, unchanged; EXL3_*/VLLM_EXL3_*/THOR_*/NCCL_* from the caller's environment pass through, and so do
# MALLOC_*, GLIBC_TUNABLES and PYTORCH_CUDA_ALLOC_CONF (host heap and CUDA allocator settings).
set -u
G=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${ENV_FILE:-$G/beam-gb10.env}"
: "${NODE_RANK:?set NODE_RANK=0 (head, $HEAD_HOST) or 1 (worker, $WORKER_HOST)}"
if [[ $NODE_RANK == 0 ]]; then MY_IP=$HEAD_IP; else MY_IP=$WORKER_IP; fi
ip -br addr show "$NET_IF" | grep -q "$MY_IP/" || { echo "NODE_RANK=$NODE_RANK expects $MY_IP on $NET_IF of this host ($(hostname))"; exit 1; }
if [[ $NODE_RANK == 0 ]]; then
  [[ -n ${API_KEY:-} ]] || { [[ -f $API_KEY_FILE ]] || (umask 077; head -c 24 /dev/urandom | base64 | tr -d '/+=' > "$API_KEY_FILE"); API_KEY=$(cat "$API_KEY_FILE"); }
  export API_KEY
fi
[[ -f $PACK/config.json ]] || { echo "no config.json in PACK=$PACK"; exit 1; }
[[ $SPEC != dflash || -f $DRAFT/config.json ]] || { echo "DFlash2 drafter missing at DRAFT=$DRAFT (or SPEC=none)"; exit 1; }
for m in ${XMOE_EXT:-} xmoe_do8 x8rh; do
  [[ -d $RECIPE/kernels/build/$m ]] || { echo "kernels/build/$m missing: run gb10/build-kernels.sh"; exit 1; }
done
[[ -z $HOSTPLAN || -f $PACK/$HOSTPLAN ]] || { echo "$PACK/$HOSTPLAN missing"; exit 1; }
mkdir -p "$CACHE_DIR"
# Runtime knobs (read by patches/vllm/sitecustomize.py or the patched vllm-exl3 plugin), as upstream serve.sh sets them.
[[ -n $HOSTPLAN ]] && export EXL3_HOST_EXPERTS=${EXL3_HOST_EXPERTS:-/model/$HOSTPLAN}
export EXL3_HOST_OVERLAP_SMS=${EXL3_HOST_OVERLAP_SMS:-32} EXL3_HOST_OVERLAP_SMS_PREFILL=${EXL3_HOST_OVERLAP_SMS_PREFILL:-32}
export EXL3_HOST_OVERLAP_GROUP=${EXL3_HOST_OVERLAP_GROUP:-16}
export EXL3_FUSED_PREP=${EXL3_FUSED_PREP:-1} EXL3_PREP_IO=${EXL3_PREP_IO:-1}
export EXL3_FUSED_DENSE_ROWS=${EXL3_FUSED_DENSE_ROWS:-32}
export EXL3_XDENSE=${EXL3_XDENSE:-1}
export EXL3_DO=${EXL3_DO:-1} EXL3_DO_MOD=${EXL3_DO_MOD:-xmoe_do8} EXL3_DO_INT8=${EXL3_DO_INT8:-1}
export EXL3_DO_DIR=${EXL3_DO_DIR:-/opt/recipe/kernels/build/$EXL3_DO_MOD}
export EXL3_FP8_PROJ=${EXL3_FP8_PROJ:-1} EXL3_FP8_DTYPE=${EXL3_FP8_DTYPE:-int8}
export EXL3_X8_EXT_DIR=${EXL3_X8_EXT_DIR:-/opt/recipe/kernels/build/x8rh} EXL3_FP8_EXT_DIR=${EXL3_FP8_EXT_DIR:-/opt/recipe/kernels/build/f8rh}
export THOR_EMBED_HOST
export THOR_LMHEAD_FP8=${THOR_LMHEAD_FP8:-1} THOR_LMHEAD_DTYPE=${THOR_LMHEAD_DTYPE:-int8}
export THOR_DRAFT_INT8=${THOR_DRAFT_INT8:-1}
export THOR_RSCALE_TOPK=${THOR_RSCALE_TOPK:-1}
export THOR_SWA=${THOR_SWA:-1} THOR_SWA_MAX_SEQS=${THOR_SWA_MAX_SEQS:-$MAX_NUM_SEQS}
extra=()
[[ $EAGER == 1 ]] && extra+=(--enforce-eager) || extra+=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"]}' --cudagraph-capture-sizes $CAPTURE_SIZES)
[[ -n $NO_CUSTOM_AR ]] && extra+=(--disable-custom-all-reduce)
[[ $KV_OFFLOAD != 0 ]] && extra+=(--kv-offloading-size "$KV_OFFLOAD")
[[ -n ${KV_MEM:-} ]] && extra+=(--kv-cache-memory "$KV_MEM")
[[ $SPEC == dflash ]] && extra+=(--speculative-config '{"method":"dflash","model":"/draft","num_speculative_tokens":'"$SPEC_K"',"attention_backend":"'"$SPEC_ATTN"'","enable_adaptive_verification":false}')
if [[ $NODE_RANK == 0 ]]; then
  extra+=(--host "$API_HOST" --port "$PORT" --api-key "$API_KEY" --served-model-name "$SERVED")
else
  extra+=(--headless)
fi
PASS_ENV=$(env | grep -E "^(VLLM_EXL3_[A-Z0-9_]+|EXL3_[A-Z0-9_]+|THOR_[A-Z0-9_]+|NCCL_[A-Z0-9_]+|VLLM_[A-Z0-9_]+|BEAM_GB10_[A-Z0-9_]+|MALLOC_[A-Z0-9_]+|GLIBC_TUNABLES|PYTORCH_CUDA_ALLOC_CONF)=" | grep -v "^VLLM_EXL3_NO_CUDA=" | sed "s/^/-e /" | tr "\n" " ")
docker rm -f "$NAME" >/dev/null 2>&1
SITE=/usr/local/lib/python3.12/dist-packages   # beam_gb10.pth there imports beam_gb10_hooks.py (GB10 load hook) in every process
# Expandable segments stay off (PYTORCH_CUDA_ALLOC_CONF empty), as upstream.
docker run -d --name "$NAME" --gpus all --network host --ipc host --shm-size 32g \
  --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PACK":/model:ro -v "$DRAFT":/draft:ro -v "$RECIPE":/opt/recipe:ro -v "$CACHE_DIR":/root/.cache \
  -v "$RECIPE/patches/vllm/sitecustomize.py":/usr/lib/python3.12/sitecustomize.py:ro \
  -v "$G/beam_gb10_hooks.py":$SITE/beam_gb10_hooks.py:ro -v "$G/beam_gb10.pth":$SITE/beam_gb10.pth:ro \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e PYTORCH_CUDA_ALLOC_CONF= ${API_KEY:+-e API_KEY} \
  -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  -e VLLM_HOST_IP="$MY_IP" -e GLOO_SOCKET_IFNAME="$NET_IF" -e NCCL_SOCKET_IFNAME="$NET_IF" -e NCCL_IB_HCA="$NCCL_IB_HCA" \
  -e NCCL_DEBUG="$NCCL_DEBUG" -e NCCL_MAX_NCHANNELS="$NCCL_MAX_NCHANNELS" ${GLIBC_TUNABLES:+-e GLIBC_TUNABLES=$GLIBC_TUNABLES} \
  ${XMOE_EXT:+-e XMOE_EXT=$XMOE_EXT -e XMOE_DIR=/opt/recipe/kernels/build} $PASS_ENV \
  "$IMAGE" /model \
  --tensor-parallel-size 2 --nnodes 2 --node-rank "$NODE_RANK" --master-addr "$HEAD_IP" --master-port "$MASTER_PORT" \
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_BATCHED" \
  --gpu-memory-utilization "$UTIL" --kv-cache-dtype "$KV_DTYPE" --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser beam --reasoning-parser beam \
  "${extra[@]}" ${EXTRA_ARGS:-}
