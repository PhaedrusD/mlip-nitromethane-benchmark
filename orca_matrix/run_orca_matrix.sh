#!/bin/bash
# Resumable ORCA driver, same as run_all.sh: logs each job to <job>.out,
# skips finished ones. Set ORCA if orca is not on PATH:
#     ORCA=/opt/orca/orca bash run_orca_matrix.sh
# On the cluster the install is not on PATH until orca_env.sh is sourced.
[ -f "$HOME/orca_env.sh" ] && source "$HOME/orca_env.sh"
ORCA=${ORCA:-${ORCA_BIN:-orca}}
command -v "$ORCA" >/dev/null || { echo "orca not found: source ~/orca_env.sh or set ORCA"; exit 1; }
cd "$(dirname "$0")"
ok () { grep -q "ORCA TERMINATED NORMALLY" "$1" 2>/dev/null; }
run () {
    ok "$1.out" && { echo "skip $1"; return 0; }
    echo "=== $1  $(date +%H:%M:%S)"
    "$ORCA" "$1.inp" > "$1.out" 2>&1
    ok "$1.out" || { echo "    FAILED: see $1.out"; return 1; }
}

for f in $(ls -S -r *_opt.inp 2>/dev/null); do run "${f%.inp}"; done

for f in $(ls -S -r *_sp.inp 2>/dev/null); do
    b=${f%_sp.inp}
    if [ -f "${b}_opt.inp" ] && [ ! -s "${b}_opt.xyz" ]; then
        echo "SKIP ${b}_sp: no ${b}_opt.xyz (optimisation failed?)"
        continue
    fi
    run "${b}_sp"
done

echo; echo "done. Check:  grep -L \"TERMINATED NORMALLY\" *.out"
