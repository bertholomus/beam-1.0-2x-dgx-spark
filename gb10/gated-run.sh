#!/bin/bash
# gated-run.sh: final-run.sh's decode rows (with the DFlash2 acceptance), prefill and TTFT rows for a lane that takes real
# traffic: every row runs alone through gb10/gated_rows.py, only while the server has no other requests, and is run again if one overlapped.
#   systemd-run --user --unit beam-rows bash $HOME/beam-serve/gb10/gated-run.sh <tag>
# Writes ~/beam-serve/rows/{rows-decode,rows-prefill,accept}-<tag>.json (each row with its gate record); log ~/beam-serve/logs/bench-<tag>.out.
TAG=${1:?tag}
DEC=${DEC:-decode1_prose_2k,decode1_code_2k,decode1_prose_46k,decode1_code_46k,decode4_code_2k,decode4_code_46k,decode4_prose_46k}
PRE=${PRE:-prefill_prose_8k,prefill_code_8k,prefill_prose_46k,prefill_code_46k,ttft_short,ttft_behind_cold46k}
B=$HOME/beam-serve; mkdir -p $B/rows $B/logs; exec >> $B/logs/bench-$TAG.out 2>&1
echo "== $(date -u +%FT%TZ) gated decode rows $TAG"
env ROWS=$DEC OUT=$B/rows/rows-decode-$TAG.json ACCEPT_OUT=$B/rows/accept-$TAG.json python3 $B/gb10/gated_rows.py
echo "== $(date -u +%FT%TZ) gated prefill + TTFT rows $TAG"
env ROWS=$PRE OUT=$B/rows/rows-prefill-$TAG.json python3 $B/gb10/gated_rows.py
echo "== $(date -u +%FT%TZ) BENCH_DONE $TAG"
