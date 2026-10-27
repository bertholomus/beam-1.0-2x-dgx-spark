#!/bin/bash
# get-kld-base.sh: fetch AtomicChat's llama.cpp BF16 KLD base for Beam (neutral set, 133 x 4096) into ~/beam-serve/kl,
# 8 range connections per part, then check the HF LFS sha256 of each part.
set -u
mkdir -p ~/beam-serve/kl && cd ~/beam-serve/kl
U=https://huggingface.co/datasets/AtomicChat/Beam-1.0-501B-A23B-GGUF-metrics/resolve/main/kld
python3 ~/beam-serve/gb10/hfrange.py $U/base-neutral.kld.00.part base-neutral.kld.00.part 48318382080 8 > dl-00.log 2>&1 &
p0=$!
python3 ~/beam-serve/gb10/hfrange.py $U/base-neutral.kld.01.part base-neutral.kld.01.part 34415430892 8 > dl-01.log 2>&1 &
p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
echo "download rc $r0 $r1"
cat > base-neutral.sha256 <<'S'
11dbb6be8a7b5e5a0e4cff98622c5f848b697823070e329fcc915ea579f281b4  base-neutral.kld.00.part
efa87ef8f9564b6fb2ea281adb9053b19e9cedc074d6dbf62b97b0c5b20a57a9  base-neutral.kld.01.part
S
sha256sum -c base-neutral.sha256 && echo KLD_BASE_OK || echo KLD_BASE_BAD
