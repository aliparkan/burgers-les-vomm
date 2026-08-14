#!/usr/bin/env bash
set -euo pipefail
mkdir -p outputs/validation outputs/figures
python src/burgers_les.py simulate --config configs/validation/moin_d2_ab2.json
python src/burgers_les.py simulate --config configs/validation/moin_d4_rk4.json
python src/postprocess.py moin \
  --d2 outputs/validation/moin_d2_ab2.npz \
  --d4 outputs/validation/moin_d4_rk4.npz \
  --outdir outputs/figures
