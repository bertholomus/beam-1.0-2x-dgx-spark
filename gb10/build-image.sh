#!/usr/bin/env bash
# build-image.sh: GB10 port of recipe/docker/build.sh (diffbot's Beam-1.0-501B-A23B-EXL3 recipe).
# Builds the serving image beam-exl3-gb10 on one GB10 node (aarch64, sm_121, CUDA 13.0):
#   Reflection's Beam vLLM image, arm64 build (vllm/vllm-openai:reflection-beam-arm64-cu130, pinned by digest)
#   + vllm-exl3 d3cfd394 with recipe/patches/vllm-exl3/exl3-beam.patch (AGPL-3.0-only; LICENSE/NOTICE copied into the image)
#   + ExLlamaV3 5be88657 built for TORCH_CUDA_ARCH_LIST=$ARCH (default 12.1a: GB10 is CC 12.1 and sm_120a cubins do not load there)
#     with gb10/exllamav3-5be8865-aarch64.patch: 5be88657's CPU MoE offload code is x86-only (AVX intrinsics, x86 pause); the patch
#     adds the same non-x86 guards and stubs upstream ExLlamaV3 v1.6.5 has (CPU offload unused here; every GPU kernel unchanged).
# Changes vs upstream build.sh: base image (arm64 digest), ARCH (12.0a -> 12.1a), the aarch64 guard patch, tag, AGPL files kept in
# /usr/share/doc/vllm-exl3. Same refs, same beam patch, same steps otherwise. Uses `docker run` + `docker commit` like upstream.
set -euo pipefail
R=${RECIPE:?set RECIPE to the recipe folder from the diffbot repo}
G=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TAG=${TAG:-beam-exl3-gb10}
BASE=${BASE:-vllm/vllm-openai:reflection-beam-arm64-cu130@sha256:7fc940745c389123cf06d150f0dc5022b7d7dafa5e958426248f46b4442b198d}
ARCH=${ARCH:-12.1a}
VLLM_EXL3_REF=d3cfd394920360d69f820d2dc96f8292a9e10283
EXLLAMAV3_REF=5be886578ec80324c2c715269387be2058724b6e
C=${BUILD_CONTAINER:-beam-img-build}; docker rm -f "$C" >/dev/null 2>&1 || true
docker run --name "$C" --memory="${BUILD_MEM:-64g}" --cpus="${BUILD_CPUS:-$(nproc)}" --entrypoint bash \
  -e VLLM_EXL3_REF=$VLLM_EXL3_REF -e EXLLAMAV3_REF=$EXLLAMAV3_REF -e MAX_JOBS="${MAX_JOBS:-16}" -e ARCH="$ARCH" \
  -v "$R/patches":/mnt/patches:ro -v "$G":/mnt/gb10:ro "$BASE" -lc '
set -euxo pipefail
apt-get update && apt-get install -y --no-install-recommends git patch libcusparse-dev-13-0 libcusolver-dev-13-0
git clone https://github.com/vcruz305/vllm-exl3.git /opt/vllm-exl3 && git -C /opt/vllm-exl3 checkout "$VLLM_EXL3_REF"
cd /opt/vllm-exl3 && patch -p1 < /mnt/patches/vllm-exl3/exl3-beam.patch
VLLM_EXL3_NO_CUDA=1 python3 -m pip install --no-build-isolation --no-deps /opt/vllm-exl3
mkdir -p /usr/share/doc/vllm-exl3 && cp /mnt/patches/vllm-exl3/LICENSE /mnt/patches/vllm-exl3/NOTICE.vllm-exl3 \
  /mnt/patches/vllm-exl3/THIRD_PARTY_NOTICES.vllm-exl3.md /mnt/patches/vllm-exl3/README.md /mnt/patches/vllm-exl3/exl3-beam.patch \
  /usr/share/doc/vllm-exl3/
git clone https://github.com/turboderp-org/exllamav3.git /opt/exllamav3 && git -C /opt/exllamav3 checkout "$EXLLAMAV3_REF"
git -C /opt/exllamav3 apply /mnt/gb10/exllamav3-5be8865-aarch64.patch && git -C /opt/exllamav3 diff --stat
TORCH_CUDA_ARCH_LIST=$ARCH python3 -m pip install --no-build-isolation --no-deps /opt/exllamav3
python3 -c "import torch, exllamav3, exllamav3_ext, vllm_exl3; from vllm_exl3 import exl3; assert hasattr(exllamav3_ext, \"exl3_moe\"); print(\"imports ok\")"
cuobjdump --list-elf $(python3 -c "import exllamav3_ext; print(exllamav3_ext.__file__)") | head -3
rm -rf /opt/exllamav3 /opt/vllm-exl3 /var/lib/apt/lists/*'
docker commit --change 'ENV VLLM_PLUGINS=vllm_exl3' --change 'ENTRYPOINT ["vllm","serve"]' --change 'CMD []' "$C" "$TAG" >/dev/null
docker rm "$C" >/dev/null
docker images --no-trunc --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' "$TAG"
echo BUILD_DONE
