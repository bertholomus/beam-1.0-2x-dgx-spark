#!/bin/bash
# run-quant.sh: the quantization step behind the published pack, as it ran on one DGX Spark (about 27 h of compute): diffbot's
# recipe/quantize/requant_seq.py on ExLlamaV3 v1.6.5 (commit cdc785d) in nvcr.io/nvidia/pytorch:26.07-py3, one layer at a time:
#   routed experts 3 bpw, all 256 per layer (--bits 3), plus a 2-bit copy of the 36 lowest-traffic experts per layer
#   (--cold-bits 2 with diffbot's host36-robust.json; that cold tier is not used by the served configuration), attention and
#   dense MLPs 6 bpw (--dense-bits 6), calibration calib128-tokens.npy (quantize/make-calib.sh).
# requant_seq.py resumes after the last finished layer: rerun this script after a stop. Then quantize/assemble.sh.
# Env (all required): BF16 (reflection/Beam-1.0-501B-A23B), RECIPE (the recipe/ folder of diffbot's repo), EXL3_SRC (an
# exllamav3 checkout at cdc785d), CALIB (calib128-tokens.npy), PLAN (host36-robust.json), OUT (output dir, ~200 GB),
# STATE (local dir for the residual-stream checkpoints; keep it on local disk); optional LAST (last layer, 51).
set -euo pipefail
: "${BF16:?}" "${RECIPE:?}" "${EXL3_SRC:?}" "${CALIB:?}" "${PLAN:?}" "${OUT:?}" "${STATE:?}"
LAST=${LAST:-51}; TORCH_EXT=${TORCH_EXT:-$OUT/../torch_ext}
mkdir -p "$OUT" "$STATE" "$TORCH_EXT"
[ -L "$OUT/state" ] || ln -s /beamstate "$OUT/state"   # requant_seq.py writes its checkpoints under OUT/state
docker run --rm --name beam-quant --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --network none \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/src:/recipe/quantize:/recipe/kl \
  -v "$EXL3_SRC":/src -v "$RECIPE":/recipe:ro -v "$(dirname "$CALIB")":/calib:ro -v "$PLAN":/plan.json:ro \
  -v "$BF16":/model:ro -v "$OUT":/out -v "$STATE":/beamstate -v "$TORCH_EXT":/root/.cache/torch_extensions -w /recipe/quantize \
  nvcr.io/nvidia/pytorch:26.07-py3 \
  python3 requant_seq.py --model /model --tokens "/calib/$(basename "$CALIB")" --bits 3 --cold-bits 2 --dense-bits 6 \
    --plan /plan.json --out /out --gpus 0 --last-layer "$LAST"
[ "$LAST" = 51 ] && touch "$OUT/.DONE"
echo QUANT_DONE
