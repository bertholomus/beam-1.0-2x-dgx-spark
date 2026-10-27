#!/bin/bash
# memtrace.sh: 1 s MemAvailable trace + low-memory guard for the Beam lane on this node.
# Appends "epoch MemAvailable_MiB" to ~/beam-serve/logs/mem-<host>-<tag>.log. If MemAvailable stays under GUARD_MIB (default 4096)
# for 3 s in a row it kills the $NAME container: a GB10 driven to zero free memory can wedge and then needs a power cycle.
# Run as a user unit: systemd-run --user --unit beam-mem bash ~/beam-serve/gb10/memtrace.sh <tag>
TAG=${1:-run}; GUARD_MIB=${GUARD_MIB:-4096}; NAME=${NAME:-beam-exl3}
L=$HOME/beam-serve/logs/mem-$(hostname)-$TAG.log; low=0
while true; do
  m=$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)
  echo "$(date +%s) $m" >> "$L"
  if [ "$m" -lt "$GUARD_MIB" ]; then low=$((low + 1)); else low=0; fi
  if [ "$low" -ge 3 ]; then
    echo "$(date +%s) GUARD: MemAvailable ${m} MiB < ${GUARD_MIB} MiB for 3 s, docker kill $NAME" >> "$L"
    docker kill "$NAME" >/dev/null 2>&1; low=0
  fi
  sleep 1
done
