#!/bin/bash
# Resumable ORCA driver: logs each job to <job>.out, skips finished ones.
# Set your ORCA path if 'orca' is not on PATH:  ORCA=/opt/orca/orca bash run_all.sh
ORCA=${ORCA:-orca}
cd "$(dirname "$0")"
for inp in *.inp; do
  out="${inp%.inp}.out"
  if grep -q "ORCA TERMINATED NORMALLY" "$out" 2>/dev/null; then
    echo "skip $inp (already done)"; continue
  fi
  echo "=== $inp  $(date) ==="
  "$ORCA" "$inp" > "$out"
done
echo "all done $(date)"
