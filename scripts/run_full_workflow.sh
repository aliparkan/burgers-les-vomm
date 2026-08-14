#!/usr/bin/env bash
set -euo pipefail

# 1) Lightweight numerical validation
python src/burgers_les.py simulate --config configs/validation/moin_d2_ab2.json
python src/burgers_les.py simulate --config configs/validation/moin_d4_rk4.json
python src/postprocess.py moin   --d2 outputs/validation/moin_d2_ab2.npz   --d4 outputs/validation/moin_d4_rk4.npz   --outdir outputs/figures

# 2) Production DNS reference and filtered LES training window
python src/burgers_les.py simulate --config configs/dns/reference_d4_rk4.json

# 3) Re-train VOMM coefficients (optional; production data required)
python src/burgers_les.py train-vomm --config configs/training/d2_ab2/stage1_bound035.json
python src/burgers_les.py train-vomm --config configs/training/d2_ab2/stage2_bound050.json
python src/burgers_les.py train-vomm --config configs/training/d2_ab2/stage3_bound070.json
python src/burgers_les.py train-vomm --config configs/training/d4_rk4/stage1_bound035.json
python src/burgers_les.py train-vomm --config configs/training/d4_rk4/stage2_bound050.json

# 4) Production LES comparisons
python src/burgers_les.py simulate --config configs/les/d2_ab2/no_model.json
python src/burgers_les.py simulate --config configs/les/d2_ab2/dsm.json
python src/burgers_les.py simulate --config configs/les/d2_ab2/vomm.json
python src/burgers_les.py simulate --config configs/les/d4_rk4/no_model.json
python src/burgers_les.py simulate --config configs/les/d4_rk4/dsm.json
python src/burgers_les.py simulate --config configs/les/d4_rk4/vomm.json

# 5) Example D4+RK4 comparison figures
python src/postprocess.py profile   --dns outputs/dns/dns_n8192_d4_rk4_t200.npz   --les outputs/les/les_n512_no_model_d4_rk4_t200.npz outputs/les/les_n512_dsm_d4_rk4_t200.npz outputs/les/les_n512_vomm_d4_rk4_t200.npz   --outdir outputs/figures
python src/postprocess.py kinetic   --dns outputs/dns/dns_n8192_d4_rk4_t200.npz   --les outputs/les/les_n512_no_model_d4_rk4_t200.npz outputs/les/les_n512_dsm_d4_rk4_t200.npz outputs/les/les_n512_vomm_d4_rk4_t200.npz   --outdir outputs/figures
python src/postprocess.py spectrum   --dns outputs/dns/dns_n8192_d4_rk4_t200.npz   --les outputs/les/les_n512_no_model_d4_rk4_t200.npz outputs/les/les_n512_dsm_d4_rk4_t200.npz outputs/les/les_n512_vomm_d4_rk4_t200.npz   --outdir outputs/figures --tmin 100 --tmax 200
