#!/bin/bash
# report.sh: memory report for one boot on this node: boot stages, idle before the stress, lowest during it.
#   bash ~/beam-serve/gb10/report.sh <tag>      (reads the running container's log, or logs/container-<tag>-<host>.log)
tag=$1; h=$(hostname); B=$HOME/beam-serve/logs; M=$B/mem-$h-$tag.log; I=$B/meminfo-$h-$tag.log
L=$B/container-$tag-$h.log; [[ -f $L ]] || { L=$(mktemp); docker logs --timestamps beam-exl3 > "$L" 2>&1; }
at() { awk -v t="$1" '$1>=t{print; exit}' "$I"; }
ev() { local ts e; ts=$(tr '\r' '\n' < "$L" | grep -m1 -E "$1" | cut -c1-19); [[ -z $ts ]] && return
  e=$(date -u -d "${ts}Z" +%s); printf '%-18s %s  %s\n' "$2" "${ts:11}" "$(at $e | cut -d' ' -f2-)"; }
echo "== $h $tag   columns: avail free cached shmem anon gpu api core worker (MiB)"
echo "first sample       $(head -1 $I | cut -d' ' -f2-)"
ev "non-default args" "api start"; ev "Loading weights took" "weights loaded"; ev "Model loading took" "model loaded"
ev "reserved .* memory for KV Cache" "KV reserved"; ev "Graph capturing finished" "graphs captured"; ev "Application startup complete" "API ready"
s=$(grep -m1 "stress $tag" $B/bench-$tag.out 2>/dev/null | awk '{print $2}'); [[ -n $s ]] && { e=$(date -u -d "$s" +%s); echo "idle before stress $(at $((e-1)) | cut -d' ' -f2-)"; }
ts0=$(tr '\r' '\n' < "$L" | grep -m1 'non-default args' | cut -c1-19); e0=$(date -u -d "${ts0}Z" +%s)   # this boot only
low=$(awk -v a=$e0 'NF>=10 && $1>=a' "$I" | sort -k2 -n | head -1); echo "lowest avail       $(echo $low | cut -d' ' -f2-)  at +$(( $(echo $low | cut -d' ' -f1) - e0 )) s after api start"
echo "max gpu/api/worker $(awk -v a0=$e0 'NF>=10 && $1>=a0{if($7>g)g=$7; if($8>a)a=$8; if($10>w)w=$10} END{print g, a, w}' "$I")"
echo "guard trace lowest $(sort -k2 -n $M 2>/dev/null | head -1 | cut -d' ' -f2) MiB; $(grep -c GUARD $M 2>/dev/null) guard kills"
tr '\r' '\n' < "$L" | grep -E "Model loading took|GPU KV cache size|Graph capturing finished|int8 weight-only \(|Initial free memory" | sed -E 's/^.*\] //' | cut -c1-150 | sort -u
