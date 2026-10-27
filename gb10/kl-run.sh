#!/bin/bash
# kl-run.sh: served KL of the running Beam lane vs AtomicChat's llama.cpp BF16 base on eval_neutral (133 x 4096),
# with the recipe's kl/served_kld.py (top-128 prompt log-probs, lower bound of the full KL).
# Runs on a node whose 127.0.0.1:$PORT reaches the API (the head node, or the relay node), with SPEC=none on the lane. Env: LABEL, CHUNKS (133 = full set), CONC (2), TOPK (128).
set -euo pipefail
LABEL=${LABEL:-served}; CHUNKS=${CHUNKS:-133}; CONC=${CONC:-2}; TOPK=${TOPK:-128}; PORT=${PORT:-8000}
K=$HOME/beam-serve/kl
API_KEY=$(cat $HOME/beam-serve/api_key); export API_KEY   # passed by name, so the key never shows in the process list
docker run --rm --network host -v "${RECIPE:?set RECIPE to the recipe folder from the diffbot repo}":/r:ro -v $K:/kl -e API_KEY \
  --entrypoint python3 beam-exl3-gb10 /r/kl/served_kld.py --port $PORT --model beam-1.0 --tokens /kl/neutral-tokens.npy \
  --base-kld /kl/base-neutral.kld.00.part /kl/base-neutral.kld.01.part --topk $TOPK --chunks $CHUNKS --conc $CONC \
  --json /kl/served-kld-$LABEL.json --label "$LABEL"
