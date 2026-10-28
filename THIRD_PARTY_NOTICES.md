# Third-party notices

The scripts in `gb10/`, `tf/` and `quantize/` are MIT (`LICENSE`). Everything below keeps its own license.

| Component | Upstream | License | In this repository |
|---|---|---|---|
| Beam EXL3 recipe (Diffbot) | [diffbot/Beam-1.0-501B-A23B-EXL3-3bpw-2x-RTX-PRO-6000](https://huggingface.co/diffbot/Beam-1.0-501B-A23B-EXL3-3bpw-2x-RTX-PRO-6000), folder `recipe/`, revision `cf1bdadc806b1a29a2e4a17e0628c7efb2cc0e27` | MIT, Copyright (c) 2026 Diffbot; its `patches/vllm-exl3/` is AGPL-3.0-only | `gb10/build-image.sh`, `gb10/build-kernels.sh` and `gb10/serve-node.sh` are ports of its `docker/build.sh`, `kernels/build.sh` and `serve/serve.sh`; `gb10/rows.py` follows the methods of its `bench/` scripts. `quantize/` runs its `quantize/requant_seq.py`, `make_cold_pack.py`, `make_pack.py` and `kl/make_tokens.py`; `gb10/kl-run.sh` runs its `kl/` scorer. The recipe itself is downloaded, not copied here, except `third_party/vllm-exl3/` (next row). Diffbot's notice is below. |
| vllm-exl3 Beam patch | the recipe's `patches/vllm-exl3/`, for [vcruz305/vllm-exl3](https://github.com/vcruz305/vllm-exl3) @ `d3cfd394` | AGPL-3.0-only, with the MIT and Apache notices in `NOTICE.vllm-exl3` and `THIRD_PARTY_NOTICES.vllm-exl3.md` | `third_party/vllm-exl3/`: `exl3-beam.patch`, `LICENSE` (AGPL-3.0 text), `NOTICE.vllm-exl3`, `THIRD_PARTY_NOTICES.vllm-exl3.md` and `README.md`, byte for byte as in the recipe at the revision above. `gb10/build-image.sh` applies the recipe's copy of the patch (the same bytes) and keeps these files in the image under `/usr/share/doc/vllm-exl3/`. |
| ExLlamaV3 | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) @ `5be88657` (serving image), v1.6.5 `cdc785d` (quantization) | MIT, Copyright (c) 2025 Turboderp | `gb10/exllamav3-5be8865-aarch64.patch`: our patch for aarch64 builds, with ExLlamaV3 context lines and guards that mirror upstream v1.6.5. Turboderp's notice is below. |
| TensorFold | [ashhart/TensorFold](https://github.com/ashhart/TensorFold); served from our fork [bertholomus/TensorFold](https://github.com/bertholomus/TensorFold), branch `beam-1.0-tp2` @ `d00201b4` | Apache-2.0 (ashhart and the TensorFold contributors) | Nothing copied. `tf/beam-tf-serve.sh` copies one commit of your clone of the fork to the nodes at each start; the fork carries TensorFold's license and `THIRD_PARTY_NOTICES.md`, with a section on Beam-1.0. |
| vLLM, Reflection's Beam vLLM image | [vllm-project/vllm](https://github.com/vllm-project/vllm), `vllm/vllm-openai:reflection-beam-arm64-cu130` | Apache-2.0 | Nothing copied. `gb10/beam_gb10_hooks.py` (ours) changes vLLM's KV-cache grouping and safetensors loading at runtime. The image is the build base and is not included. |
| Beam-1.0-501B-A23B, Beam-1.0-501B-A23B-DFlash2 (Reflection) | [reflection/Beam-1.0-501B-A23B](https://huggingface.co/reflection/Beam-1.0-501B-A23B), [reflection/Beam-1.0-501B-A23B-DFlash2](https://huggingface.co/reflection/Beam-1.0-501B-A23B-DFlash2) | Apache-2.0, (c) Reflection | Not in this repository. Our EXL3 quant is [bertholomus/Beam-1.0-501B-A23B-EXL3-3bpw-2xGB10](https://huggingface.co/bertholomus/Beam-1.0-501B-A23B-EXL3-3bpw-2xGB10) (Apache-2.0, with Reflection's LICENSE). The drafter is downloaded from Reflection. |
| KL reference logits and calibration text (AtomicChat) | [AtomicChat/Beam-1.0-501B-A23B-GGUF-metrics](https://huggingface.co/datasets/AtomicChat/Beam-1.0-501B-A23B-GGUF-metrics) | see the dataset | Not included. `gb10/get-kld-base.sh` (with `gb10/hfrange.py`) and `quantize/make-calib.sh` use files you download from it. |

---

## Diffbot, Beam EXL3 recipe

```
MIT License

Copyright (c) 2026 Diffbot

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Turboderp, ExLlamaV3

```
MIT License

Copyright (c) 2025 Turboderp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
