#!/bin/bash
# assemble.sh: builds the servable pack from run-quant.sh's layer packs with diffbot's recipe/quantize/make_cold_pack.py and
# make_pack.py (non-expert tensors from the BF16 release, dense layers 6 bpw, tokenizer and chat template from the BF16 release).
# Env (all required): BF16, RECIPE, EXL3_SRC, PLAN (as for run-quant.sh), OUT (run-quant.sh's output), PACK (the pack to write).
set -euo pipefail
: "${BF16:?}" "${RECIPE:?}" "${EXL3_SRC:?}" "${PLAN:?}" "${OUT:?}" "${PACK:?}"
mkdir -p "$(dirname "$PACK")"
docker run --rm --name beam-pack --network none -e PYTHONPATH=/src \
  -v "$EXL3_SRC":/src -v "$RECIPE":/recipe:ro -v "$BF16":/model:ro -v "$OUT":/q -v "$PLAN":/plan.json:ro \
  -v "$(dirname "$PACK")":/models -w /recipe/quantize nvcr.io/nvidia/pytorch:26.07-py3 bash -c "
    set -e
    ls /q/cold/config.json >/dev/null 2>&1 || python3 make_cold_pack.py /q/cold 2
    python3 make_pack.py --fp8 /model --default /q --dense /q --dense-bits 6 --cold /q/cold --cold-plan /plan.json \
      --tokenizer-dir /model --out /models/$(basename "$PACK")"
du -sh "$PACK"
echo ASSEMBLED
