#!/bin/bash
# memguard.sh CONTAINER FLOOR_MIB LOG [STOPFILE]: a 1 s MemAvailable trace in LOG; stops CONTAINER when MemAvailable falls
# under FLOOR_MIB. Exits when CONTAINER is gone or STOPFILE exists. tf/beam-tf-serve.sh starts one beside each container.
C=$1; FLOOR=$2; LOG=$3; STOP=${4:-$LOG.stop}
low=999999999
while :; do
  [ -e "$STOP" ] && { echo "$(date -u +%FT%TZ) stopfile, lowest $low MiB" >> $LOG; exit 0; }
  docker inspect -f '{{.State.Running}}' "$C" 2>/dev/null | grep -q true || { echo "$(date -u +%FT%TZ) $C not running, lowest $low MiB" >> $LOG; exit 0; }
  m=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  [ "$m" -lt "$low" ] && low=$m
  echo "$(date -u +%FT%TZ) $m" >> $LOG
  if [ "$m" -lt "$FLOOR" ]; then echo "$(date -u +%FT%TZ) GUARD $m < $FLOOR MiB: docker stop $C" >> $LOG; docker stop -t 5 "$C" >/dev/null 2>&1; exit 1; fi
  sleep 1
done
