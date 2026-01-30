#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT="${SCRIPT:-scripts/train_classifier.py}"

TASK="${TASK:-genre}"
DATA_ROOT="${DATA_ROOT:-data/sample}"
OUT_ROOT="${OUT_ROOT:-outputs_test}"

BACKBONE="${BACKBONE:-mobilenetv3small}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
BATCH_SIZE="${BATCH_SIZE:-16}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-1}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-1}"

STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-50}"
VAL_STEPS="${VAL_STEPS:-10}"

"${PYTHON_BIN}" "${SCRIPT}" \
  --task "${TASK}" \
  --data_root "${DATA_ROOT}" \
  --out_root "${OUT_ROOT}" \
  --backbone "${BACKBONE}" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --stage1_epochs "${STAGE1_EPOCHS}" \
  --stage2_epochs "${STAGE2_EPOCHS}" \
  --steps_per_epoch "${STEPS_PER_EPOCH}" \
  --validation_steps "${VAL_STEPS}" \
  --repeat
