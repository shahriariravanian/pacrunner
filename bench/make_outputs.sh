#!/bin/bash
# make_outputs.sh: the benchmark outputs in BENCH (default WORK/outputs, WORK default ~/pacrunner-bench; an existing
# output is kept): make_output.py (test_mg7_amplicol's card: fixed scales, NNPDF23, seed 7), the benchmark card
# (all threads, 64k batches, 100k events, the colour below, device cpu) and a first stock run, which builds the
# stock library. bench.py makes the NAME_gpu copies for cuda-full. MG7 (or PACRUNNER_MADGRAPH7), RUNPY: run.sh's.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-$HOME/pacrunner-bench}
BENCH=${BENCH:-$WORK/outputs}
export MG7=${MG7:-${PACRUNNER_MADGRAPH7:?MG7 or PACRUNNER_MADGRAPH7: the MadGraph7 checkout}}
RUNPY=${RUNPY:-python3}
mkdir -p "$BENCH/logs"
card() {  # card DIR COLOUR
  local f=$1/Cards/run_card.toml
  sed -e 's/^cpu_thread_pool_size = .*/cpu_thread_pool_size = -1/' -e 's/^cpu_batch_size = .*/cpu_batch_size = 64000/' \
      -e 's/^max_batch_size = .*/max_batch_size = 64000/' -e 's/^events = .*/events = 100000/' \
      -e "s/^color_accuracy = .*/color_accuracy = \"$2\"/" -e 's/^device = .*/device = ["cpu"]/' "$f" > "$f.new"
  mv "$f.new" "$f"
}
while read -r name colour proc; do
  D=$BENCH/$name
  if [ ! -d "$D" ]; then
    t=$(date +%s)
    "$RUNPY" "$HERE/make_output.py" "$D" "$proc" < /dev/null > "$BENCH/logs/$name-output.log" 2>&1
    echo "$name: output in $(( $(date +%s) - t )) s"
  fi
  card "$D" "$colour"
  if ls "$D"/lib/libamplicol_*.so > /dev/null 2>&1; then
    echo "$name: stock library already built"
  else
    t=$(date +%s)
    (cd "$D" && "$RUNPY" bin/generate_events -f < /dev/null > "$BENCH/logs/$name-first.log" 2>&1)
    echo "$name: first stock run (builds the stock library) in $(( $(date +%s) - t )) s"
  fi
done <<LIST
pp_zjj lc p p > z j j [amplicol]
pp_jjj lc p p > j j j [amplicol]
gg_ttgg full g g > t t~ g g [amplicol]
uu_zgggg lc u u~ > z g g g g [amplicol]
gg_4g lc g g > g g g g [amplicol]
LIST
echo "outputs in $BENCH"
