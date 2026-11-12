#!/bin/bash
: "${ORCA:?set ORCA to the orca binary}"
cd "$(dirname "$0")"
ok () { grep -q "ORCA TERMINATED NORMALLY" "$1" 2>/dev/null; }
run () {   # $1 = basename
    ok "$1.out" && { echo "skip $1"; return 0; }
    echo "=== $1  $(date +%H:%M:%S)"
    "$ORCA" "$1.inp" > "$1.out" 2>&1
    ok "$1.out" || { echo "    FAILED: see $1.out"; return 1; }
}

# PASS 1: optimisations (smallest first, so failures surface fast)
for f in $(ls -S -r *_opt.inp 2>/dev/null); do run "${f%.inp}"; done

# PASS 2: single points. Each needs its _opt.xyz, so it is checked
# for -- a missing geometry means the optimisation failed and the
# coupled-cluster job would silently use the wrong structure.
for f in $(ls -S -r *_sp.inp 2>/dev/null); do
    b=${f%_sp.inp}
    if [ -f "${b}_opt.inp" ] && [ ! -s "${b}_opt.xyz" ]; then
        echo "SKIP ${b}_sp: no ${b}_opt.xyz (optimisation failed?)"
        continue
    fi
    run "${b}_sp"
done

echo; echo "done. Check for failures:  grep -L \"TERMINATED NORMALLY\" *.out"
