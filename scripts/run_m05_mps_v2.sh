#!/usr/bin/env bash
# Re-run M0.5 heavy phases with realistic training budget.
# Phase A already passed in the v1 run — skipping.
set -u
cd "$(dirname "$0")/.."
OUT="${OUT:-runs/m05_mps_v2}"
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
log "M0.5 v2 pipeline starting (proper sizing) | OUT=$OUT MODEL=$MODEL DEV=$DEV"

# Capacity sweep: way more epochs so the model actually trains
run_phase "3_capacity" $PY scripts/factoid_capacity_sweep.py \
  --model "$MODEL" --out "$OUT" --device "$DEV" \
  --n-list 64,128,256,512,1024 \
  --batch 8 --seq-len 1024 --epochs 60 --lr 1e-3 --repeats 80

# Phase B: heavy healing pass + real eval
run_phase "4_phaseB" $PY scripts/run_phase_b_gates.py \
  --model "$MODEL" --out "$OUT" --device "$DEV" \
  --n-facts 256 --batch 8 --seq-len 1024 --epochs 40 --lr 1e-3 \
  --n-partitions 8 --train-repeats 60

# Phase C (HEADLINE G6): the most critical run gets the most compute
run_phase "5_phaseC" $PY scripts/run_knowledge_displacement.py \
  --model "$MODEL" --out "$OUT" --device "$DEV" \
  --n-facts 150 --batch 8 --seq-len 1024 --epochs 80 --lr 1e-3 \
  --train-repeats 80 --skip-g7

log "M0.5 v2 pipeline COMPLETE — aggregating gates"
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
    if 'acc_weights_path' in g6[-1]:
        print(f\"    weights-path acc = {g6[-1].get('acc_weights_path'):.3f}\")
        print(f\"    mount-path acc   = {g6[-1].get('acc_mount_path'):.3f}\")
        print(f\"    control (nomount) = {g6[-1].get('acc_mount_path_nomount_control'):.3f}\")
" 2>&1 | tee -a "$LOG"
