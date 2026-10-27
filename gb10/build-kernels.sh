#!/usr/bin/env bash
# build-kernels.sh: GB10 port of recipe/kernels/build.sh. Same four modules and build_ext.py, built inside beam-exl3-gb10
# for TORCH_CUDA_ARCH_LIST=$ARCH (default 12.1a; upstream 12.0a) into recipe/kernels/build/<name>/ (no GPU needed).
#   xmoe_dense  decode MoE kernel + flat dense decode GEMM          (XMOE_EXT=xmoe_dense, EXL3_XDENSE=1)
#   xmoe_do8    decode-once prefill MoE GEMM, INT8 tensor cores      (EXL3_DO=1, EXL3_DO_INT8=1)
#   x8rh        INT8 prefill projections (weights reconstructed once) (EXL3_FP8_PROJ=1, EXL3_FP8_DTYPE=int8)
#   f8rh        the e4m3 variant (EXL3_FP8_DTYPE=fp8)
set -euo pipefail
K=${RECIPE:?set RECIPE to the recipe folder from the diffbot repo}/kernels; IMAGE=${IMAGE:-beam-exl3-gb10}; ARCH=${ARCH:-12.1a}
for V in ${@:-xmoe_dense xmoe_do8 x8rh f8rh}; do
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e TORCH_CUDA_ARCH_LIST=$ARCH -e MAX_JOBS=${MAX_JOBS:-16} \
    -v "$K":/k -w /k --entrypoint python3 "$IMAGE" build_ext.py "$V"
done
for V in ${@:-xmoe_dense xmoe_do8 x8rh f8rh}; do
  echo "$V: $(docker run --rm -v "$K":/k --entrypoint cuobjdump "$IMAGE" --list-elf /k/build/$V/$V.so | sed 's/.*\.\(sm_[0-9a-z]*\)\..*/\1/' | sort | uniq -c | tr '\n' ' ')"
done
echo KERNELS_DONE
