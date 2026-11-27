#!/bin/bash
# run.sh SUITE [PACRUNNER [TAG]] [-- bench.py options]: bench.py's suite (full or cpu) with pacrunner from PACRUNNER
# (default: this repository), a git checkout (its HEAD, copied by git archive to WORK/pacrunner-<commit>/, so
# that a run names one commit) or a copied tree with a COMMIT file. Logs in WORK/logs/<suite>-<commit>[-TAG]/
# (machine.txt, bench.out, summary.txt). bench_full.sh and bench_cpu.sh call this.
#
# Environment:
#   WORK    the working folder (default ~/pacrunner-bench)
#   BENCH   the outputs, NAME and NAME_gpu (default WORK/outputs: make_outputs.sh)
#   MG7     MadGraph7 with MadSpace (madspace/install-cuda for cuda-full); default PACRUNNER_MADGRAPH7
#   RUNPY   the outputs' python, MadGraph7's (default python3)
#   PY      the python that builds PacRunner's libraries (default RUNPY)
#   SYMJIT  a symjit Python package folder to put first on PYTHONPATH (default: the installed symjit)
#   REPEATS pass A's runs per variant (default 3)
# Symbolica (pyAmpliCol) needs a writable XDG_RUNTIME_DIR.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SUITE=${1:?SUITE (full or cpu)}; shift
SRC=$(dirname "$HERE"); TAG=
if [ $# -gt 0 ] && [ "$1" != "--" ]; then SRC=$(realpath "$1"); shift; fi
if [ $# -gt 0 ] && [ "$1" != "--" ]; then TAG=$1; shift; fi
[ "${1:-}" = "--" ] && shift
WORK=${WORK:-$HOME/pacrunner-bench}
BENCH=${BENCH:-$WORK/outputs}
MG7=${MG7:-${PACRUNNER_MADGRAPH7:?MG7 or PACRUNNER_MADGRAPH7: the MadGraph7 checkout}}
RUNPY=${RUNPY:-python3}
PY=${PY:-$RUNPY}
SYMJIT=${SYMJIT:-}
[ -d "$BENCH" ] || { echo "no outputs in $BENCH: run make_outputs.sh first" >&2; exit 1; }
mkdir -p "$WORK/logs"
if git -C "$SRC" rev-parse --git-dir > /dev/null 2>&1; then
  COMMIT=$(git -C "$SRC" rev-parse --short HEAD)
  [ -z "$(git -C "$SRC" status --porcelain --untracked-files=no)" ] || echo "note: $SRC has uncommitted changes; benchmarking its HEAD $COMMIT"
  PAC=$WORK/pacrunner-$COMMIT
  [ -d "$PAC" ] || { mkdir "$PAC"; git -C "$SRC" archive HEAD | tar -x -C "$PAC"; echo "$COMMIT" > "$PAC/COMMIT"; }
else
  PAC=$SRC; COMMIT=$(cat "$PAC/COMMIT")
fi
L=$WORK/logs/$SUITE-$COMMIT${TAG:+-$TAG}; mkdir -p "$L"
export PACRUNNER_MADGRAPH7=$MG7 PACRUNNER_PYTHON=$PY
[ -d /usr/local/cuda/bin ] && export PATH=/usr/local/cuda/bin:$PATH
export PYTHONPATH=${SYMJIT:+$SYMJIT:}$PAC
{ date; uname -srm; (lscpu 2>/dev/null | grep 'Model name' || sysctl -n machdep.cpu.brand_string) | sed 's/  */ /g'
  command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
  command -v nvcc > /dev/null && nvcc --version | tail -1 || true
  echo "MadGraph7 $(git -C "$MG7" log --oneline -1)"; echo "pacrunner $COMMIT"
  echo "symjit $("$PY" -c 'import symjit, os; print(getattr(symjit, "__version__", ""), os.path.dirname(symjit.__file__))')" \
       "$( [ -n "$SYMJIT" ] && git -C "$SYMJIT" log --oneline -1 2>/dev/null || true)"
} > "$L/machine.txt" 2>&1
"$PY" "$HERE/bench.py" --suite "$SUITE" --bench "$BENCH" --logs "$L" --runpy "$RUNPY" --python "$PY" --pacrunner "$PAC" \
  ${REPEATS:+--repeats $REPEATS} "$@" 2>&1 | tee "$L/bench.out"
"$PY" "$HERE/summary.py" "$L" > "$L/summary.txt" 2>&1
echo "summary: $L/summary.txt"
