#!/bin/bash
# ─────────────────────────────────────────────────────────
#  Run all AD-MT ablation experiments sequentially
#  Usage: ./run_ablation.sh [GPU_ID]
# ─────────────────────────────────────────────────────────
set -e

GPU_ID=${1:-0}
SCRIPT="./run_yolo_admt.sh"

echo "╔══════════════════════════════════════════════════╗"
echo "║      AD-MT Ablation Study — YOLO Detection      ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Full AD-MT (baseline for comparison)
echo ">>> [1/5] Full AD-MT (RPA + CCM + student arbitration)"
bash ${SCRIPT} ${GPU_ID} config_yolo_admt.yml
echo ""

# Ablation A: 1-teacher self-training baseline
echo ">>> [2/5] Ablation: 1-teacher self-training (no 2-teacher RPA/CCM)"
bash ${SCRIPT} ${GPU_ID} ablation/ablation_1teacher.yml
echo ""

# Ablation B: No CCM
echo ">>> [3/5] Ablation: No CCM (conflict_weight=0)"
bash ${SCRIPT} ${GPU_ID} ablation/ablation_no_ccm.yml
echo ""

# Ablation C: CCM teacher-only
echo ">>> [4/5] Ablation: CCM teacher-only (no student arbitration)"
bash ${SCRIPT} ${GPU_ID} ablation/ablation_ccm_tea_only.yml
echo ""

# Ablation D: Fixed period RPA
echo ">>> [5/5] Ablation: Fixed period RPA"
bash ${SCRIPT} ${GPU_ID} ablation/ablation_fixed_period.yml
echo ""

echo "╔══════════════════════════════════════════════════╗"
echo "║            All 5 ablations completed!             ║"
echo "║  Check results/ for mAP50, loss curves, etc.    ║"
echo "╚══════════════════════════════════════════════════╝"
