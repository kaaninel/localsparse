#!/usr/bin/env bash
# Composite local smoke runner (plan §7.7).
# Validates that every M1.5 script runs end-to-end without crashing,
# using tiny budgets so the whole thing finishes in a few minutes on CPU.
# Exit non-zero on any failure — must pass before pushing.

set -e
cd "$(dirname "$0")/.."

echo "=== [1/4] pytest ==="
python -m pytest -q

echo ""
echo "=== [2/4] gemma4 local smoke ==="
python scripts/gemma4_local_smoke.py

echo ""
echo "=== [3/4] bench_veyra3 dry-run all sections ==="
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-float32}"
RUN_DIR="/tmp/local_smoke_$$"
mkdir -p "$RUN_DIR"
for s in a0 a1 a2 a3 a4 a5 a6 a7 b0 b1 b2 b3 a8; do
  echo "  -- section $s --"
  python scripts/bench_veyra3.py \
    --section "$s" --device "$DEVICE" --dtype "$DTYPE" \
    --dry-run --run_dir "$RUN_DIR/$s" >"$RUN_DIR/$s.log" 2>&1 || {
      echo "FAIL: section $s; last lines:"
      tail -20 "$RUN_DIR/$s.log"
      exit 1
    }
  echo "    ok"
done

echo ""
echo "=== [4/4] all checks PASS ==="
echo "Ready to push and run Colab notebook notebooks/m15_unified_colab.ipynb"
