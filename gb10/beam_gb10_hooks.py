"""GB10 hooks for the Beam lane, loaded in every Python process of the lane by beam_gb10.pth.

1. Load hook (on unless BEAM_GB10_CLONE_LOAD=0). On GB10 unified memory, a host-to-GPU copy that reads straight from a
memory-mapped safetensors file is pathologically slow: the GPU faults the file pages in one at a time (UVM bottom half busy,
GPU idle, the copying thread spinning). vLLM's lazy loader hands out such mmap views, and the vllm-exl3 row-parallel loader
(o_proj, down_proj) copies a contiguous narrow of one straight into the device parameter; our first boot spent 200-434 s on single
dense-Lxx files and rank 1 sat 17 min in layers.0.mlp.down_proj. Measured on one Spark: 11.8 MB o_proj trellis half, mmap direct
15 ms in a fresh process, minutes inside the loaded worker; clone to RAM first, then copy, 4 ms.
This hook makes safe_open(...).get_tensor() return a copy in ordinary RAM (CPU tensors only), so every later GPU copy reads
present anonymous pages. Extra RAM at any moment = one tensor (the largest is the ~1.6 GB vocab table). Weights are unchanged
bit for bit.

2. KV cache group size (BEAM_GB10_KV_GROUP_SIZE, unset = vLLM's choice). vLLM splits the KV layers into groups as large as its
smallest attention type and pads the last group of each type. With the DFlash2 drafter that is the drafter's 5 window-4096
layers, so Beam's 9 full-attention layers take 2 groups with one padding layer: 10 x 1,024 B of KV per token per rank instead
of 9 x 1,024 B. BEAM_GB10_KV_GROUP_SIZE=9 puts the 9 full-attention layers in one group (the 43 window-512 layers in 5, the
drafter's 5 in 1). Memory layout only: every layer keeps its own KV spec, block size and values; the rest is vLLM's code.

3. Heap trim (BEAM_GB10_TRIM_S=<seconds>, unset = off): a daemon thread calls glibc malloc_trim(0) every <seconds>, which
hands free heap pages back to the OS (glibc only returns the top of each arena by itself).
"""
import os

_QUIET = os.environ.get("BEAM_GB10_HOOK_QUIET", "0") == "1"

if os.environ.get("BEAM_GB10_CLONE_LOAD", "1") != "0":
    try:
        import safetensors

        _safe_open = safetensors.safe_open

        class safe_open:  # noqa: N801  (drop-in for safetensors.safe_open)
            def __init__(self, *args, **kwargs):
                self._h = _safe_open(*args, **kwargs)

            def __enter__(self):
                self._h.__enter__()
                return self

            def __exit__(self, *exc):
                return self._h.__exit__(*exc)

            def get_tensor(self, name):
                t = self._h.get_tensor(name)
                return t.clone() if getattr(t, "is_cuda", True) is False else t

            def __getattr__(self, attr):
                return getattr(self._h, attr)

        safetensors.safe_open = safe_open
        if not _QUIET:
            print("[beam-gb10] safetensors get_tensor -> RAM copy (GB10 load hook)", flush=True)
    except Exception as e:  # never block the server on this hook
        print(f"[beam-gb10] load hook not installed: {e}", flush=True)

_KV_GROUP = os.environ.get("BEAM_GB10_KV_GROUP_SIZE", "")
if _KV_GROUP:
    try:
        import importlib.abc
        import sys
        from collections import defaultdict

        _KV_PLANNING = "vllm.v1.core.kv_cache_planning"

        def _patch_kv_planning(P):
            size = int(_KV_GROUP)

            def _get_kv_cache_groups_uniform_page_size(kv_cache_spec):
                # vLLM's function, with its group size (smallest type; the largest when within 1.5x) replaced by `size`.
                same_type_layers = defaultdict(list)
                for layer_name, layer_spec in kv_cache_spec.items():
                    same_type_layers[layer_spec].append(layer_name)
                layer_buckets, spec_buckets = [], []
                for layer_spec, layer_names in same_type_layers.items():
                    for names, specs in zip(layer_buckets, spec_buckets):
                        try:
                            type(specs[0]).merge([*specs, layer_spec])
                        except (AssertionError, ValueError):
                            continue
                        names.extend(layer_names)
                        specs.append(layer_spec)
                        break
                    else:
                        layer_buckets.append(list(layer_names))
                        spec_buckets.append([layer_spec])
                grouped_layers = []
                for layers in layer_buckets:
                    num_groups = -(-len(layers) // size)
                    grouped_layers.extend(layers[i::num_groups] for i in range(num_groups))
                if not _QUIET:
                    print(f"[beam-gb10] KV groups of {size}: types {[len(b) for b in layer_buckets]} -> "
                          f"{len(grouped_layers)} groups {[len(g) for g in grouped_layers]}", flush=True)
                return P.create_kv_cache_group_specs(kv_cache_spec, grouped_layers)

            P._get_kv_cache_groups_uniform_page_size = _get_kv_cache_groups_uniform_page_size

        class _KVPlanningHook(importlib.abc.MetaPathFinder):
            # Patches vllm.v1.core.kv_cache_planning right after it is first imported (vLLM is not imported yet at .pth time).
            def find_spec(self, fullname, path, target=None):
                if fullname != _KV_PLANNING:
                    return None
                for finder in sys.meta_path:
                    if finder is self or not hasattr(finder, "find_spec"):
                        continue
                    spec = finder.find_spec(fullname, path, target)
                    if spec is not None:
                        break
                else:
                    return None
                exec_module = spec.loader.exec_module

                def _exec(module):
                    exec_module(module)
                    _patch_kv_planning(module)

                spec.loader.exec_module = _exec
                return spec

        sys.meta_path.insert(0, _KVPlanningHook())
    except Exception as e:
        print(f"[beam-gb10] KV group hook not installed: {e}", flush=True)

_TRIM_S = os.environ.get("BEAM_GB10_TRIM_S", "")
if _TRIM_S:
    try:
        import ctypes
        import threading
        import time

        _malloc_trim = ctypes.CDLL("libc.so.6").malloc_trim

        def _trim_loop():
            while True:
                time.sleep(float(_TRIM_S))
                _malloc_trim(0)

        def _start_trim():
            threading.Thread(target=_trim_loop, name="beam-gb10-trim", daemon=True).start()

        _start_trim()
        os.register_at_fork(after_in_child=_start_trim)
    except Exception as e:
        print(f"[beam-gb10] heap trim not started: {e}", flush=True)
