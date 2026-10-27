# Beam-1.0-501B-A23B on two DGX Spark

**Served weights:** [bertholomus/Beam-1.0-501B-A23B-EXL3-3bpw-2xGB10](https://huggingface.co/bertholomus/Beam-1.0-501B-A23B-EXL3-3bpw-2xGB10),
our EXL3 quant (Apache-2.0).
**Base model:** [reflection/Beam-1.0-501B-A23B](https://huggingface.co/reflection/Beam-1.0-501B-A23B) by Reflection, a US lab
(Apache-2.0). Model and weights (c) Reflection.

This repository holds the scripts that build, serve, check and quantize that EXL3 quant on two NVIDIA DGX Spark (GB10) nodes:
vLLM tensor parallel 2 over one 200 Gb/s link, Reflection's DFlash2 drafter, and a 655,360-token context window. Everything runs
on the two machines. It is a GB10 port of Diffbot's Beam EXL3 recipe for 2x RTX PRO 6000 (MIT): the quantizer, the kernels and
the vllm-exl3 Beam patch are Diffbot's; the GB10 ports, the memory work and the checks here are ours.

## What we ran

| | |
|---|---|
| Nodes | 2 x NVIDIA DGX Spark (GB10, 128 GB unified memory, aarch64) |
| Link | QSFP port 0 of each node (`enp1s0f0np0`, 200 Gb/s, MTU 9000), RoCE on both of its functions (`rocep1s0f0`, `roceP2p1s0f0`), through a 200G switch. A direct QSFP cable between the two nodes was not tested. |
| Image | `beam-exl3-gb10` (38 GiB) from `gb10/build-image.sh`: Reflection's arm64 Beam vLLM image (pinned by digest), vllm-exl3 `d3cfd394` with the recipe's Beam patch, ExLlamaV3 `5be88657` built for sm_121 |
| Kernels | the recipe's `xmoe_dense`, `xmoe_do8`, `x8rh` and `f8rh`, built for sm_121 by `gb10/build-kernels.sh` |
| Weights | routed experts 3 bits (all 256 per layer), attention and dense MLPs 6 bits, embedding, norms, router and lm_head BF16: ~90 GiB per node |
| Drafter | [reflection/Beam-1.0-501B-A23B-DFlash2](https://huggingface.co/reflection/Beam-1.0-501B-A23B-DFlash2), 3 draft tokens per step |
| Window | 655,360 tokens, fp8 KV cache, 5.91 GiB KV pool per node (vLLM: 656,887 tokens), up to 8 sequences |
| API | OpenAI-compatible, served name `beam-1.0`, Bearer key, chat with reasoning and tool calls (Beam's parsers) |

Rank 0 (engine core, API server, TP rank 0) runs on one node, rank 1 (headless worker) on the other.

## Quick start

1. **Nodes.** Docker with the NVIDIA runtime, user systemd with lingering (`loginctl enable-linger`), and passwordless ssh
   from your control machine to both nodes (`spark1`, `spark2` below). Give QSFP port 0 of each node an address on one subnet
   (`10.0.0.1` and `10.0.0.2` below); we run MTU 9000.

2. **Files**, at the same paths on both nodes (the same user and home directory on both). Download on one node and copy to
   the other over the 200G link. We ship no drafter weights: download them from Reflection.

   ```bash
   hf download bertholomus/Beam-1.0-501B-A23B-EXL3-3bpw-2xGB10 --local-dir ~/models/Beam-1.0-501B-A23B-EXL3
   hf download reflection/Beam-1.0-501B-A23B-DFlash2 --local-dir ~/models/Beam-1.0-501B-A23B-DFlash2
   hf download diffbot/Beam-1.0-501B-A23B-EXL3-3bpw-2x-RTX-PRO-6000 --revision cf1bdadc806b1a29a2e4a17e0628c7efb2cc0e27 \
     --include 'recipe/*' --local-dir ~/beam-serve/dl && mv ~/beam-serve/dl/recipe ~/beam-serve/recipe
   ```

   Keep the recipe at `~/beam-serve/recipe`: the bench scripts look there unless `RECIPE` or `CORPUS` is set.

3. **Build** on one node (with this repository checked out there), then copy to the other:

   ```bash
   export RECIPE=$HOME/beam-serve/recipe
   gb10/build-image.sh                                   # docker image beam-exl3-gb10
   gb10/build-kernels.sh                                 # $RECIPE/kernels/build/{xmoe_dense,xmoe_do8,x8rh,f8rh}
   docker save beam-exl3-gb10 | ssh spark2 docker load
   ```

   Copy `$RECIPE/kernels/build/` to the same place on the other node, or run `gb10/build-kernels.sh` there too.
   `gb10/beam-serve.sh up` refuses to start when the image differs between the nodes or a kernel module is missing.

4. **Serve**, from the control machine:

   ```bash
   H=$(ssh spark1 'echo $HOME')   # the nodes' home directory
   export PACK=$H/models/Beam-1.0-501B-A23B-EXL3 DRAFT=$H/models/Beam-1.0-501B-A23B-DFlash2 RECIPE=$H/beam-serve/recipe
   export HEAD_HOST=spark1 HEAD_IP=10.0.0.1 WORKER_HOST=spark2 WORKER_IP=10.0.0.2 API_HOST=10.0.0.1 API_FWD_HOST=spark2
   gb10/beam-serve.sh up       # copies gb10/ to ~/beam-serve/gb10 on both nodes, preflight, rank 1, rank 0, memory traces, relay
   gb10/beam-serve.sh status   # containers, /v1/models, KV pool, MemAvailable now and lowest since up
   gb10/beam-serve.sh down
   ```

   Our last start was ready after 157 s. Every setting is in
   `gb10/beam-gb10.env` and can be set for one run, e.g. `SPEC=none gb10/beam-serve.sh up`.

5. **Use.** The API listens only on the head node's link address (`http://10.0.0.1:8000/v1`); the other node relays its
   `127.0.0.1:8000` to it (`gb10/apifwd.py`). The key is `~/beam-serve/api_key` on the head node, made at the first start
   (mode 600). Copy it to the relay node for clients there.

   ```bash
   curl -s http://127.0.0.1:8000/v1/chat/completions -H "Authorization: Bearer $(cat ~/beam-serve/api_key)" \
     -H 'Content-Type: application/json' -d '{"model": "beam-1.0", "messages": [{"role": "user", "content": "Hello"}]}'
   ```

6. **Check**, on the relay node: `systemd-run --user --unit beam-bench bash $HOME/beam-serve/gb10/final-run.sh <tag>` runs
   the five checked prompts, the longest prompt, the decode rows with the drafter's acceptance, and the prefill and TTFT rows
   (results in `~/beam-serve/{probe,rows,logs}`). On a server that already has users, `gb10/gated-run.sh <tag>` runs the same
   rows one at a time, only while no other request is in the server, and runs a row again if another request overlapped it.

## Measured

All numbers are ours, from the scripts in `gb10/`: vLLM's OpenAI API over HTTP, temperature 0, DFlash2 with 3 draft tokens,
no other requests in the server. Medians of 3 runs, with the range in brackets.

### Speed

| Row | Prompt tokens | Result |
|---|---|---|
| Decode, 1 stream, prose | 2,062 | 28.9 tok/s [28.1-30.2] |
| Decode, 1 stream, code | 1,520 | 34.2 tok/s [32.4-35.0] |
| Decode, 1 stream, prose | 46,007 | 29.6 tok/s [26.8-30.3] |
| Decode, 1 stream, code | 35,697 | 32.3 tok/s [29.9-32.8] |
| Decode, 4 streams, code, aggregate | ~2K each | 60.3 tok/s [58.4-65.0] |
| Decode, 4 streams, code, aggregate | ~36K each | 55.0 tok/s [52.9-55.0] |
| Decode, 4 streams, prose, aggregate | ~46K each | 54.6 tok/s [52.8-57.8] |
| Cold prefill, prose | 8,058-8,083 | 1,411 tok/s [1,403-1,411], first token 5.73 s |
| Cold prefill, code | 6,098-7,119 | 1,306 tok/s [1,290-1,360], first token 5.45 s |
| Cold prefill, prose | 45,926-45,978 | 1,371 tok/s [1,370-1,373], first token 33.5 s |
| Cold prefill, code | 35,316-37,768 | 1,305 tok/s [1,300-1,314], first token 27.4 s |
| Short chat, first token, idle server | short | 0.371 s [0.369-0.465] (5 runs) |
| Short chat, first token, behind a cold 46K prefill | short | 32.7 s (1 run) |
| DFlash2 mean acceptance length, over the 7 decode rows | | 2.762 (draft acceptance 0.587) |

Methods (`gb10/rows.py`, after the recipe's `bench/` scripts): decode tok/s = 384 / (t400 - t16) from pairs of 16 and 400 new
tokens (ignore_eos, reasoning effort low) after warming the prompt's prefix cache; 4 streams = 4 different prompts sent together,
4 x 384 / (t400 - t16); cold prefill = fresh prompts (no prefix-cache hit), max_tokens 1, prompt tokens / wall time; first
token = streaming, time to the first delta. Prose prompts are a word-soup document plus a writing task; code prompts are real
code from the recipe's `bench/corpus.txt` plus "explain what this code does".

These rows were measured with a 400,000-token window, before the four memory changes that the 655,360 default needs (KV
groups of 9, 4 NCCL channels, 128 MiB FlashInfer workspaces, rank 0 and the API on the node with more free memory; see below).
With the shipped defaults, the same fixed prompts in the same order (2 runs each) measured 28.9 tok/s (decode, 1 stream,
prose 46K), 60.9 tok/s aggregate (decode, 4 streams, code 2K) and 1,260 tok/s (cold prefill, code 6.2K and 6.8K), against 29.2,
60.7 and 1,268 tok/s before those changes.

A long prompt holds up the others: vLLM's chunked prefill gives it the whole 2,048-token step budget, so a short chat that
arrives behind a 46K prefill waits for it (the 32.7 s row), and one behind a 632K prompt would wait minutes.

### Longest prompt

| Window | Prompt + new tokens | Time | Average prefill |
|---|---|---|---|
| 655,360 (shipped) | 632,185 + 8 | 755.7 s | 837 tok/s |
| 400,000 | 399,913 + 8 | 409.2 s | 977 tok/s |

One run each (`gb10/maxctx.py`): real code from the recipe's corpus behind a unique first line, so nothing comes from the
prefix cache; prefill time = time to the first streamed token. The 632K run filled 96.5 % of the window because that version of
`maxctx.py` ran out of text; the one here sizes the prompt to the window.

### Quality

| KL mean (lower bound) | KL median | KL q99 | Top-1 agreement | Perplexity ratio vs BF16 |
|---|---|---|---|---|
| 0.013200 | 0.000982 | 0.1768 | 96.663 % | 1.0125 |

KL divergence against the BF16 model on AtomicChat's `eval_neutral` set: all 133 chunks x 4,096 tokens (272,251 scored
positions), the served model's prompt log-probs (top 128 per position) through the vLLM API, against AtomicChat's BF16
reference logits, scored with the recipe's `kl/served_kld.py`. KL over the top-128 support is a lower bound of the full KL.
Two runs on two configurations (MAX_BATCHED 4,096 with NCCL's 64 channels, and MAX_BATCHED 2,048 with 8 channels):
KL 0.013179 and 0.013200, top-1 96.666 % and 96.663 %, perplexity ratio 1.0124 and 1.0125.

To repeat it: `gb10/get-kld-base.sh` (82.7 GB of reference logits), the recipe's `kl/make_tokens.py --tokenizer
$PACK/tokenizer.json --neutral-kld ~/beam-serve/kl/base-neutral.kld.00.part --out ~/beam-serve/kl`, a lane started with
`SPEC=none MAX_MODEL_LEN=8192 KV_MEM=536870912 EXTRA_ARGS="--max-logprobs 128"` (prompt log-probs come from the target model),
then `gb10/kl-run.sh` on the relay node.

`gb10/probe5.py`: five checked prompts (2 prose with a fact to find, 2 code whose functions must pass tests in a throwaway
container with no network, 1 tool call), temperature 0, every reply with reasoning: 5/5 with the shipped defaults.

## Memory

We keep MemAvailable at or above 9 GiB on both nodes under load, which our OS image needs for everything outside the lane.
`gb10/memtrace.sh` traces it every second on each node, and `gb10/beam-serve.sh status` shows the lowest since the start.
With the shipped defaults, over the 632K prompt and all of the rows above, the lowest was 9,486 MiB on the head node (while the
API server tokenized a 655,360-token prompt) and 9,977 MiB on the worker.

The head node sets the window: it holds rank 0 plus the engine core (~0.8 GB) and the API server (~1.1 GB, more while it
tokenizes a long prompt). The next 64K step would need 576 MiB more KV on each node, which takes the head node below the floor.
Beam itself allows 1,048,576 tokens.

On an OS image that keeps more or less memory free, change MAX_MODEL_LEN and KV_MEM together:
KV_MEM = (ceil(MAX_MODEL_LEN / 16) + 2,058) x 147,456 bytes per node, which is 9,216 bytes (~0.86 GiB per 100K tokens) plus a
fixed 0.28 GiB for the sliding-window layers and the drafter. vLLM refuses to start when one full-length request does not fit.

### What the GB10 defaults change

Values in `gb10/beam-gb10.env`; the recipe's `serve/serve.sh` (2x RTX PRO 6000) in brackets.

| Setting | GB10 value | Why |
|---|---|---|
| MAX_MODEL_LEN | 655,360 [131,072] | the longest that kept 9 GiB free on both nodes with the drafter on |
| KV_MEM | 6,343,262,208 B [profiled from UTIL] | a fixed pool: on unified memory each rank profiles differently |
| BEAM_GB10_KV_GROUP_SIZE | 9 [vLLM: 5, set by the drafter's 5 layers] | Beam's 9 full-attention layers fill one KV group with no padding layer: 9,216 instead of 10,240 B per token per node |
| NCCL_MAX_NCHANNELS | 4 [NCCL's choice, 64 here] | each channel has host buffers (no GPU Direct RDMA on GB10): frees ~3.3 GiB per node at the same speed and bits |
| VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE | 128 MiB [394 MiB per builder] | above vLLM's own prefill estimate at MAX_BATCHED 2048; frees ~0.6 GiB per node |
| MAX_BATCHED | 2,048 [4,096] | ~0.5 GiB more free under load and a smaller per-request sliding-window KV, for ~14 % slower cold prefill |
| GLIBC_TUNABLES | fixed mmap and trim thresholds [glibc default] | hands freed load buffers back to the OS (~0.3 GiB per node) |
| UTIL | 0.77 [0.94] | only vLLM's start-up free-memory check here; the pool is KV_MEM |
| HOSTPLAN, THOR_EMBED_HOST, KV_OFFLOAD | none, 0, 0 [host31-robust.json, 1, 64 GiB] | host-RAM experts, the host embedding table and a CPU KV tier save nothing on unified memory |
| NO_CUSTOM_AR | 1 [one-shot P2P all-reduce] | two hosts have no P2P: NCCL all-reduce |
| API_HOST | the head node's link address [0.0.0.0] | the API is reachable on the link and through the relay's loopback only |

`gb10/beam_gb10_hooks.py` (loaded through `beam_gb10.pth`) also makes safetensors reads copy each tensor to RAM before the GPU
copy: on unified memory a host-to-GPU copy straight from a memory-mapped file is pathologically slow. Weights are unchanged bit
for bit.

## Quantize it yourself (optional)

The published weights came from these steps on one DGX Spark, in `nvcr.io/nvidia/pytorch:26.07-py3`:

1. `quantize/make-calib.sh`: 128 windows of 4,096 calibration tokens from AtomicChat's `calib_train.txt` with the recipe's
   `kl/make_tokens.py`. Our `calib128-tokens.npy` has sha256 `040d2d966f7d202f055335549e9b14cd62dec6eac4e1a3445ab3ca8729ee5a44`.
2. `quantize/run-quant.sh`: the recipe's `requant_seq.py` on ExLlamaV3 v1.6.5 (`cdc785d`), one layer at a time (about 27 h);
   it resumes after the last finished layer.
3. `quantize/assemble.sh`: the recipe's `make_cold_pack.py` and `make_pack.py` write the servable folder.

The steps also write `cold/` and `cold-plan.json` (a 2-bit tier of each layer's 36 least-used experts). This configuration
does not load them, and the published weights leave them out.

## Credits

- **Reflection**: Beam-1.0-501B-A23B, the DFlash2 drafter and the Beam vLLM image (Apache-2.0).
- **Diffbot**: the Beam EXL3 recipe this repository ports (MIT): `requant_seq.py`, the kernels, the vllm-exl3 Beam patch
  (AGPL-3.0-only) and the bench and KL methods.
- **vcruz305**: vllm-exl3 (AGPL-3.0-only). The Beam patch source and the AGPL text are in `third_party/vllm-exl3/`.
- **turboderp**: ExLlamaV3 (MIT); our aarch64 build patch is `gb10/exllamav3-5be8865-aarch64.patch`.
- **The vLLM project**: vLLM (Apache-2.0).
- **AtomicChat**: the BF16 reference logits and the calibration text.

Bertholomus AI is not affiliated with or endorsed by Reflection, Diffbot, NVIDIA, the vLLM project, vcruz305, turboderp or
AtomicChat.

## License

The scripts in this repository are MIT (`LICENSE`). `third_party/vllm-exl3/` is AGPL-3.0-only. See `NOTICE` and
`THIRD_PARTY_NOTICES.md`. The model weights are Apache-2.0, (c) Reflection.
