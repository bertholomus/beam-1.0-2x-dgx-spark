#!/bin/bash
# final-run.sh: the served-config check, as one user unit on a node whose 127.0.0.1:8000 reaches the API: probe5, maxctx at the served max_model_len, the decode
# rows with the DFlash2 mean acceptance length from the server's spec-decode counters (/metrics, before vs after), then the prefill
# and TTFT rows.   systemd-run --user --unit beam-bench bash $HOME/beam-serve/gb10/final-run.sh <tag>
# Writes ~/beam-serve/probe/<tag>/, ~/beam-serve/rows/{maxctx,rows-decode,rows-prefill,accept}-<tag>.json, log ~/beam-serve/logs/bench-<tag>.out.
TAG=${1:?tag}
DEC=${DEC:-decode1_prose_2k,decode1_code_2k,decode1_prose_46k,decode1_code_46k,decode4_code_2k,decode4_code_46k,decode4_prose_46k}
PRE=${PRE:-prefill_prose_8k,prefill_code_8k,prefill_prose_46k,prefill_code_46k,ttft_short,ttft_behind_cold46k}
B=$HOME/beam-serve; mkdir -p $B/probe/$TAG $B/rows $B/logs; exec >> $B/logs/bench-$TAG.out 2>&1
K=$(cat $B/api_key)
spec() { curl -s -m 10 -H "Authorization: Bearer $K" http://127.0.0.1:8000/metrics | awk '/^vllm:spec_decode_num_drafts(_total)?[{ ]/{d+=$NF} /^vllm:spec_decode_num_accepted_tokens(_total)?[{ ]/{a+=$NF} /^vllm:spec_decode_num_draft_tokens(_total)?[{ ]/{t+=$NF} END{printf "%d %d %d\n", d, a, t}'; }
echo "== $(date -u +%FT%TZ) probes $TAG"; OUT=$B/probe/$TAG python3 $B/gb10/probe5.py
echo "== $(date -u +%FT%TZ) maxctx $TAG"; python3 $B/gb10/maxctx.py | tee $B/rows/maxctx-$TAG.json
echo "== $(date -u +%FT%TZ) decode rows $TAG"; s0=$(spec)
env OUT=$B/rows/rows-decode-$TAG.json ROWS=$DEC REPS=${REPS:-3} python3 $B/gb10/rows.py; s1=$(spec)
python3 - "$s0" "$s1" "$B/rows/accept-$TAG.json" <<'PY'
import json, sys
d0, a0, t0 = map(int, sys.argv[1].split()); d1, a1, t1 = map(int, sys.argv[2].split())
d, a, t = d1 - d0, a1 - a0, t1 - t0
r = {"drafts": d, "accepted_tokens": a, "draft_tokens": t, "mean_acceptance_length": round(1 + a / d, 3) if d else None,
     "draft_acceptance_rate": round(a / t, 3) if t else None}
json.dump(r, open(sys.argv[3], "w")); print("acceptance over the decode rows", json.dumps(r))
PY
echo "== $(date -u +%FT%TZ) prefill + TTFT rows $TAG"
env OUT=$B/rows/rows-prefill-$TAG.json ROWS=$PRE REPS=${REPS:-3} python3 $B/gb10/rows.py
echo "== $(date -u +%FT%TZ) BENCH_DONE $TAG"
