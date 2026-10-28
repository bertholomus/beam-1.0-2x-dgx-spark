#!/bin/bash
# beam-tf-build.sh: run by tf/beam-tf-serve.sh in a throwaway container on each node before the ranks start. Builds the
# CUDA extensions, or finds them current, in $TORCH_EXTENSIONS_DIR, so no nvcc runs beside loaded weights. A lock there
# can only be a killed build's: the serve script runs this after its preflight found the node free.
find "$TORCH_EXTENSIONS_DIR" \( -name lock -o -name .ninja_lock \) -delete 2>/dev/null
pip install --no-deps --no-build-isolation -q -e /tfw/TensorFold > /w/logs/beam-tf-serve-build.pip 2>&1 || { echo "pip install failed"; exit 1; }
python3 - <<'PY' || exit 1
from tensorfold.cuda.exl3 import experts, linear
from tensorfold.cuda.kernels import qmm

for m in (linear, experts, qmm):        # the target's EXL3 layers and experts, the drafter's 4-bit weights
    m._ext()
print("BUILD_OK")
PY
