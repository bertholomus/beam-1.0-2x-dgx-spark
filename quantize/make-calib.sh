#!/bin/bash
# make-calib.sh: the calibration tokens for run-quant.sh (calib128-tokens.npy, 128 windows of 4,096 tokens) with diffbot's
# recipe/kl/make_tokens.py: AtomicChat's calib/beam-1.0-moe/calib_train.txt tokenized with Beam's tokenizer.json.
# Env (all required): BF16 (reflection/Beam-1.0-501B-A23B, for tokenizer.json), RECIPE (diffbot's recipe/ folder), CALIB_TXT
# (calib_train.txt from https://huggingface.co/datasets/AtomicChat/Beam-1.0-501B-A23B-GGUF-metrics), CALIB_DIR (output dir).
set -euo pipefail
: "${BF16:?}" "${RECIPE:?}" "${CALIB_TXT:?}" "${CALIB_DIR:?}"
mkdir -p "$CALIB_DIR"
docker run --rm --network none -v "$BF16":/model:ro -v "$RECIPE":/recipe:ro -v "$(dirname "$CALIB_TXT")":/in:ro \
  -v "$CALIB_DIR":/out nvcr.io/nvidia/pytorch:26.07-py3 bash -c "pip install -q tokenizers >/dev/null 2>&1 || true
    python3 /recipe/kl/make_tokens.py --tokenizer /model/tokenizer.json --calib /in/$(basename "$CALIB_TXT") --out /out"
ls -la "$CALIB_DIR"
