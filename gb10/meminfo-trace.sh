#!/bin/bash
# meminfo-trace.sh: 1 s memory breakdown for memory measurements (no guard; memtrace.sh is the guard).
# Appends "epoch avail free cached shmem anon gpu api core worker" (MiB; gpu = nvidia-smi compute-app memory, api/core/worker =
# RSS of the vLLM API server, EngineCore and TP worker) to ~/beam-serve/logs/meminfo-<host>-<tag>.log.
#   systemd-run --user --unit beam-meminfo bash $HOME/beam-serve/gb10/meminfo-trace.sh <tag>
TAG=${1:-run}; L=$HOME/beam-serve/logs/meminfo-$(hostname)-$TAG.log
while true; do
  m=$(awk '/^(MemAvailable|MemFree|Cached|Shmem|AnonPages):/{v[$1]=int($2/1024)} END{print v["MemAvailable:"], v["MemFree:"], v["Cached:"], v["Shmem:"], v["AnonPages:"]}' /proc/meminfo)
  g=$(timeout 3 nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
  r=$(ps -eo rss,args | awk '/vllm serve/ && !/awk/{a+=$1} /VLLM::EngineCor/{c+=$1} /VLLM::Worker/{w+=$1} END{print int(a/1024), int(c/1024), int(w/1024)}')
  echo "$(date +%s) $m $g $r" >> "$L"
  sleep 1
done
