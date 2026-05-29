#!/usr/bin/env bash
# Runs the full M0.5 pipeline on M4 Air MPS.
# Each phase writes JSONL logs under $OUT and a global log to $OUT/pipeline.log.
# Sized to fit in ~2-4 hours on M4 Air per plan §6.3.
set -u
cd "$(dirname "$0")/.."
OUT="${OUT:-runs/m05_mps}"
MODEL="${MODEL:-veyra-ai/veyra3-5m-base}"
DEV="${DEV:-mps}"
mkdir -p "$OUT"
LOG="$OUT/pipeline.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

run_phase() {
  local name="$1"; shift
  log "=== $name START ==="
  if "$@" 2>&1 | tee -a "$LOG"; then
    log "=== $name DONE ==="
  else
    log "=== $name FAILED (exit=$?) ==="
  fi
}

PY=".venv/bin/python"

log "M0.5 pipeline starting | OUT=$OUT MODEL=$MODEL DEV=$DEV"

run_phase "0_preflight"  $PY scripts/mps_preflight.py --device "$DEV" --seq-len 1024 --batch 2
run_phase "1_prebench"   $PY scripts/veyra_pre_benchmark.py --model "$MODEL" --out "$OUT" --device "$DEV" --ctx-lens 1024,2048,4096
run_phase "2_phaseA"     $PY scripts/run_phase_a_gates.py --model "$MODEL" --out "$OUT" --device "$DEV" --steps 600 --batch 2 --seq-len 512 --g3-trials 32 --g3-blocks 16
run_phase "3_capacity"   $PY scripts/factoid_capacity_sweep.py --model "$MODEL" --out "$OUT" --device "$DEV" --n-list 64,128,256,512 --epochs 5 --batch 4 --seq-len 256
run_phase "4_phaseB"     $PY scripts/run_phase_b_gates.py --model "$MODEL" --out "$OUT" --device "$DEV" --n-facts 256 --batch 4 --seq-len 384 --epochs 3 --n-partitions 8
run_phase "5_phaseC"     $PY scripts/run_knowledge_displacement.py --model "$MODEL" --out "$OUT" --device "$DEV" --n-facts 150 --batch 4 --seq-len 384 --epochs 6 --skip-g7

log "M0.5 pipeline COMPLETE — aggregating gate results"
$PY -c "
import json
from pathlib import Path
out = Path('$OUT')
rows=[]
for gp in sorted(out.glob('phase_*/gates.jsonl')):
    for ln in gp.read_text().splitlines():
        r = json.loads(ln); r['run'] = gp.parent.name
        rows.append(r)
print('\n=== FINAL GATE RESULTS ===')
for r in rows:
    mark = {'pass':'PASS','fail':'FAIL','stretch':'STRETCH','deferred':'DEFER'}[r['status']]
    print(f\"  {mark:>7s}  {r['gate_id']:<14s}  {r['metric']}={r['value']:.4f}  thr={r['threshold']:.4f}  ({r['run']})\")
g6 = [r for r in rows if r['gate_id']=='G6']
if g6:
    print()
    print('=== HEADLINE G6 ===')
    print(f\"  ratio={g6[-1]['value']:.3f}  status={g6[-1]['status'].upper()}\")
" 2>&1 | tee -a "$LOG"
