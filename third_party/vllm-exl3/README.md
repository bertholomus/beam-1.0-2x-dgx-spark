# vllm-exl3 patch (AGPL-3.0-only)

`exl3-beam.patch` applies to [vcruz305/vllm-exl3](https://github.com/vcruz305/vllm-exl3) at `d3cfd394`
(`src/vllm_exl3/exl3.py`, the only file it changes); `docker/build.sh` applies it before installing the plugin. vllm-exl3 is
licensed AGPL-3.0-only at that commit, so this patch is too (`LICENSE`); its upstream notices are in `NOTICE.vllm-exl3` and the
upstream repo's `THIRD_PARTY_NOTICES.md`, reproduced here as `THIRD_PARTY_NOTICES.vllm-exl3.md`. If you serve a modified version of the patched plugin over a network, AGPL-3.0
section 13 applies.

What it adds (each behind an environment variable; `serve/serve.sh` sets the measured configuration):

- **Host-RAM experts** (`EXL3_HOST_EXPERTS=<plan.json>`): the listed experts' trellis stays in pinned host memory (exact-size
  `cudaHostRegister`, zero-copy CUDA alias), the rest in VRAM.
- **Split launch with side-stream overlap**: VRAM experts run on the `xmoe_dense` flat decode kernel, host-RAM experts on a group
  kernel on a second stream limited to `EXL3_HOST_OVERLAP_SMS` (decode) / `EXL3_HOST_OVERLAP_SMS_PREFILL` SMs, in groups of
  `EXL3_HOST_OVERLAP_GROUP` SMs per expert (16: one expert's PCIe read 265 -> 191 us).
- **Fused routing prep** (`EXL3_FUSED_PREP=1`): one Triton launch builds the per-expert token lists for a decode call;
  `EXL3_PREP_IO=1` also casts the input to fp16 and zeroes the fp32 output in the same launch, and keeps int32 expert ids.
- **Decode-once prefill MoE** (`EXL3_DO=1`): eager calls with at least `EXL3_DO_MIN_TOKENS` (257) tokens run every VRAM expert
  through a grouped GEMM (module `EXL3_DO_MOD` in `EXL3_DO_DIR`) that decodes each 16x16 trellis tile once per job of up to 128
  rows. `EXL3_DO_INT8=1` (default in serve.sh) runs it on INT8 tensor cores: activations int8 per row x 128-wide K segment,
  decoded weights int8 at the 3-bit MCG codebook's fixed scale (127 / 3.949219). `EXL3_DO_FP8=1` is the e4m3 variant (worse KL).
- **Fused dense projections** (`EXL3_FUSED_DENSE_ROWS=32`): q/k/v/gate and the dense MLPs' gate/up run as one sliced EXL3 GEMM at
  up to 32 rows. `EXL3_XDENSE=1` replaces that, at up to 16 rows, with `xmoe_dense.xdense_forward`: one persistent launch for all
  slices, bf16 in and out (no casts, no concatenation); single-source layers (o_proj) keep the stock path unless
  `EXL3_XDENSE_SINGLE=1`.
- **INT8 prefill projections** (`EXL3_FP8_PROJ=1 EXL3_FP8_DTYPE=int8`): at load, each 6-bit attention / dense-MLP weight is
  reconstructed once into an int8 [N, K] buffer in a Hadamard-rotated basis with per-output-channel scales (`x8rh` module in
  `EXL3_X8_EXT_DIR`); prefill chunks quantize the rotated activations per token and run int8 x int8 GEMMs. `EXL3_FP8_DTYPE=fp8` is
  the e4m3 variant (`f8rh`, worse KL). Rows below `EXL3_FP8_MIN_ROWS` (auto: 256) keep the EXL3 path.
- **Routing counters** (`EXL3_EXPERT_STATS=<dir>`): per-layer routed-expert counts, split into decode-sized and prefill calls,
  dumped every `EXL3_EXPERT_STATS_EVERY` eager calls (graph-safe). They can be used to build a host-RAM plan for your own
  workload; the shipped plans come from offline calibration captures (`quantize/README.md`).
- `EXL3_FAT_SLICE=0` removes redundant 2,048-row slicing of heavily loaded experts.
- Experimental, off by default and not recommended: `EXL3_DO_COLD=1` / `EXL3_DO_COLD_FAT` (host-RAM experts through decode-once:
  slower prefill), `EXL3_HOST_DMA=1|2` (copy-engine staging of host-RAM experts in prefill), `EXL3_COLD_ON_GPU=1` with
  `EXL3_HOST_COLD_PACK` (a 2-bit cold tier in VRAM instead of 3-bit experts in host RAM: +48 % KLD).
