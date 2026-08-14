#!/usr/bin/env python3
"""
Post-processing utilities for the forced Burgers LES framework:
  1) Moin validation plot for D2/D4,
  2) dynamic coefficient versus time,
  3) u(x,t=200) comparison with filtered DNS,
  4) resolved kinetic energy time history with 1-time-unit block averaging,
  5) stationary-state energy spectrum averaged over t in [100,200].
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import iv

TWOPI = 2.0 * np.pi


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=True))


def ensure_outdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def config_from_npz(data: Dict[str, np.ndarray]) -> Dict:
    raw = data.get("config_json", None)
    if raw is None:
        return {}
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def model_label(path: str | Path, fallback: str = "LES") -> str:
    name = Path(path).stem.upper()
    if "NOMODEL" in name or "NO_MODEL" in name:
        return "No model"
    if "DSM" in name:
        return "Dynamic Smagorinsky"
    if "VOMM" in name:
        return "Adjoint-based VOMM"
    if "DNS" in name:
        return "filtered DNS"
    return fallback


def scheme_label(path: str | Path) -> str:
    """Return the spatial/time discretization encoded in an output filename."""
    name = Path(path).stem.upper()
    if "D2" in name and "AB2" in name:
        return "D2+AB2"
    if "D4" in name and "RK4" in name:
        return "D4+RK4"
    return ""


def filter_box_ps(u: np.ndarray, ratio: int) -> np.ndarray:
    ratio = int(ratio)
    if ratio <= 1:
        return np.asarray(u).copy()
    if ratio % 2 != 0:
        half = ratio // 2
        acc = np.zeros_like(u, dtype=float)
        for s in range(-half, half + 1):
            acc += np.roll(u, s)
        return acc / float(2 * half + 1)
    half = ratio // 2
    acc = 0.5 * np.roll(u, -half) + 0.5 * np.roll(u, half)
    for s in range(-half + 1, half):
        acc += np.roll(u, s)
    return acc / float(ratio)


def filter_to_les(u: np.ndarray, nx_les: int) -> np.ndarray:
    u = np.asarray(u)
    if u.size == nx_les:
        return u.copy()
    if u.size % nx_les != 0:
        raise ValueError(f"Cannot filter size {u.size} to {nx_les}; sizes are not integer multiples.")
    ratio = u.size // nx_les
    return filter_box_ps(u, ratio)[::ratio].copy()


def moin_exact_solution(x: np.ndarray, t: float, nu: float = 1.0, amp: float = 10.0, n_modes: int = 256) -> np.ndarray:
    """Cole-Hopf exact periodic solution for u(x,0)=amp*sin(x), viscous Burgers."""
    a = amp / (2.0 * nu)
    phi = iv(0, a) * np.ones_like(x, dtype=float)
    phix = np.zeros_like(x, dtype=float)
    for n in range(1, n_modes + 1):
        coef = 2.0 * iv(n, a) * np.exp(-nu * n * n * t)
        phi += coef * np.cos(n * x)
        phix += -n * coef * np.sin(n * x)
        if abs(coef) < 1.0e-14 and n > 50:
            break
    return -2.0 * nu * phix / phi


def _nearest_saved_snapshot(data: Dict[str, np.ndarray], target_time: float) -> Tuple[np.ndarray, float]:
    """Return the saved snapshot closest to target_time; t=0 is reconstructed from the IC."""
    x = np.asarray(data["x"], dtype=float)
    cfg = config_from_npz(data)
    amp = float(cfg.get("moin_amplitude", 10.0))
    if abs(target_time) < 1.0e-14:
        return amp * np.sin(x), 0.0
    t = np.asarray(data.get("t", []), dtype=float)
    if "u" not in data or t.size == 0:
        raise ValueError("Moin validation needs saved snapshots 'u' and 't'.")
    idx = int(np.argmin(np.abs(t - target_time)))
    return np.asarray(data["u"][idx], dtype=float), float(t[idx])


def plot_moin(d2: Optional[str], d4: Optional[str], outdir: str) -> Path:
    """Recreate Moin Example 6.8 / Fig. 6.10 validation.

    The figure in Moin uses u(x,0)=10 sin(x), nu=1, N=32, dt=0.005,
    periodic boundaries, and reports profiles at t=0, 0.1, 0.2, 0.4, 0.6.
    Exact profiles are obtained with the Cole-Hopf solution.
    """
    out = ensure_outdir(outdir)
    target_times = [0.0, 0.1, 0.2, 0.4, 0.6]
    datasets: List[Tuple[str, Dict[str, np.ndarray], str]] = []
    for path, label in [(d2, "D2+AB2"), (d4, "D4+RK4")]:
        if path:
            datasets.append((label, load_npz(path), path))

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    xx = np.linspace(0.0, TWOPI, 1024, endpoint=False)

    numeric_label_used = {"D2+AB2": False, "D4+RK4": False}
    for tt in target_times:
        line, = ax.plot(xx, moin_exact_solution(xx, tt), linewidth=1.8, label=f"Exact t={tt:g}")
        color = line.get_color()
        for label, data, _ in datasets:
            x = np.asarray(data["x"], dtype=float)
            u_num, t_saved = _nearest_saved_snapshot(data, tt)
            # For exact t=0 the reconstructed numerical initial condition is identical;
            # plotting it verifies that the initial condition matches the book setup.
            marker = "o" if label.startswith("D2") else "x"
            lab = label if not numeric_label_used[label] else None
            numeric_label_used[label] = True
            ax.plot(x, u_num, linestyle="none", marker=marker, markersize=3.8,
                    markeredgewidth=0.9, color=color, label=lab)
            if abs(t_saved - tt) > 5.0e-3:
                print(f"[warning] requested t={tt:g}, nearest saved snapshot for {label} is t={t_saved:g}")

    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_xlim(0.0, TWOPI)
    ax.set_title("Moin Example 6.8 / Fig. 6.10 validation")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    path = out / "validation_moin_fig6_10_D2_D4.png"
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    # Also keep the old filename expected by the README/workflow.
    legacy = out / "validation_moin_D2_D4.png"
    fig.savefig(legacy, dpi=220)
    plt.close(fig)
    return path


def plot_coefficients(files: Sequence[str], outdir: str) -> Path:
    out = ensure_outdir(outdir)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    plotted = False
    for f in files:
        data = load_npz(f)
        if "coeff" not in data or "t" not in data:
            continue
        coeff = np.asarray(data["coeff"])
        t = np.asarray(data["t"])
        label = model_label(f)
        scheme = scheme_label(f)
        if label == "Dynamic Smagorinsky" and scheme:
            label = f"Dynamic Smagorinsky - {scheme}"
        elif scheme:
            label = f"{label} - {scheme}"
        if coeff.ndim == 1 or coeff.shape[1] == 1:
            c = coeff.reshape(-1)
            ax.plot(t, c, linewidth=1.3, label=label)
            plotted = True
        else:
            for j in range(coeff.shape[1]):
                ax.plot(t, coeff[:, j], linewidth=1.3, label=f"{label} C{j+1}")
                plotted = True
    if not plotted:
        raise ValueError("No coefficient histories found in the provided files.")
    ax.set_xlabel("t")
    ax.set_ylabel("Dynamic Smagorinsky coefficient")
    ax.set_title("Dynamic Smagorinsky coefficient versus time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = out / "dynamic_coefficient_corrected.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    # Keep the legacy filename for compatibility with older workflows.
    fig.savefig(out / "dynamic_coefficients_vs_time.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_profile(dns: str, les_files: Sequence[str], outdir: str, nx_les: int = 512) -> Path:
    out = ensure_outdir(outdir)
    dns_data = load_npz(dns)
    u_dns = np.asarray(dns_data.get("u_final", dns_data["u"][-1]))
    u_fdns = filter_to_les(u_dns, nx_les)
    x_les = np.arange(nx_les) * TWOPI / nx_les
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(x_les, u_fdns, linewidth=2.0, label="filtered DNS")
    for f in les_files:
        data = load_npz(f)
        u = np.asarray(data.get("u_final", data["u"][-1]))
        ax.plot(np.asarray(data.get("x", x_les)), u, linewidth=1.1, label=model_label(f))
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\bar{u}(x,t=200)$")
    ax.set_title(r"Velocity profile comparison at $t=200$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = out / "velocity_profile_t200.png"
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def resolved_k_from_snapshots(data: Dict[str, np.ndarray], nx_les: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(data["t"], dtype=float)
    if "u" in data:
        u = np.asarray(data["u"], dtype=float)
        if u.shape[1] != nx_les:
            k = np.asarray([0.5 * np.mean(filter_to_les(row, nx_les) ** 2) for row in u])
        else:
            k = 0.5 * np.mean(u * u, axis=1)
    else:
        k = np.asarray(data["k_resolved"], dtype=float)
    return t, k


def block_average_time(t: np.ndarray, y: np.ndarray, block_length: float = 1.0, samples_per_block: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    # Project statement: blocks of length 1, using 10 independent samples separated by 0.1.
    if len(t) == 0:
        return t, y
    t0 = np.ceil(t[0] / block_length) * block_length
    tend = np.floor(t[-1] / block_length) * block_length
    tb: List[float] = []
    yb: List[float] = []
    current = t0
    while current <= tend + 1.0e-12:
        targets = current - block_length + 0.1 * np.arange(1, samples_per_block + 1)
        vals = []
        for tt in targets:
            idx = int(np.argmin(np.abs(t - tt)))
            if abs(t[idx] - tt) <= 5.0e-3:
                vals.append(y[idx])
        if vals:
            tb.append(current)
            yb.append(float(np.mean(vals)))
        current += block_length
    return np.asarray(tb), np.asarray(yb)


def plot_resolved_k(dns: str, les_files: Sequence[str], outdir: str, nx_les: int = 512) -> Path:
    out = ensure_outdir(outdir)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    dns_data = load_npz(dns)
    td, kd = resolved_k_from_snapshots(dns_data, nx_les)
    td_b, kd_b = block_average_time(td, kd)
    ax.plot(td_b, kd_b, linewidth=2.0, label="filtered DNS")
    for f in les_files:
        data = load_npz(f)
        t, k = resolved_k_from_snapshots(data, nx_les)
        tb, kb = block_average_time(t, k)
        ax.plot(tb, kb, linewidth=1.2, label=model_label(f))
    ax.set_xlabel("t")
    ax.set_ylabel(r"$k_R = 0.5\langle \bar{u}^2\rangle$")
    ax.set_title("Resolved kinetic energy, block-averaged over time intervals of length 1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = out / "resolved_kinetic_energy.png"
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def energy_spectrum_1d(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = u.size
    a = np.fft.rfft(u) / n
    k = np.fft.rfftfreq(n, d=TWOPI / n) * TWOPI
    weights = np.ones_like(k)
    if len(weights) > 2:
        weights[1:-1] = 2.0
    e = 0.5 * weights * np.abs(a) ** 2
    return k, e


def average_spectrum(data: Dict[str, np.ndarray], tmin: float, tmax: float, nx_les: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    if "u" not in data:
        raise ValueError("Spectrum averaging needs saved snapshots 'u' in the NPZ file.")
    t = np.asarray(data["t"], dtype=float)
    u = np.asarray(data["u"], dtype=float)
    mask = (t >= tmin - 1.0e-12) & (t <= tmax + 1.0e-12)
    if not np.any(mask):
        raise ValueError(f"No snapshots found in requested interval [{tmin}, {tmax}].")
    spectra = []
    k_ref = None
    for row in u[mask]:
        if row.size != nx_les:
            row = filter_to_les(row, nx_les)
        k, e = energy_spectrum_1d(row)
        k_ref = k
        spectra.append(e)
    return k_ref, np.mean(np.asarray(spectra), axis=0)


def plot_spectrum(dns: str, les_files: Sequence[str], outdir: str, tmin: float = 100.0, tmax: float = 200.0, nx_les: int = 512) -> Path:
    out = ensure_outdir(outdir)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    k, e = average_spectrum(load_npz(dns), tmin, tmax, nx_les)
    ax.loglog(k[1:], e[1:], linewidth=2.0, label="filtered DNS")
    for f in les_files:
        kk, ee = average_spectrum(load_npz(f), tmin, tmax, nx_les)
        ax.loglog(kk[1:], ee[1:], linewidth=1.2, label=model_label(f))
    ax.set_xlabel("k")
    ax.set_ylabel(r"$E(k)$")
    ax.set_title(fr"Energy spectrum averaged over $t\in[{tmin:g},{tmax:g}]$")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    path = out / "energy_spectrum_stationary.png"
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_vomm_training(train_json: str, outdir: str) -> Path:
    """Optional diagnostic for the VOMM coefficient optimization; not required as a main result."""
    out = ensure_outdir(outdir)
    with open(train_json, "r", encoding="utf-8") as f:
        d = json.load(f)
    hist = d.get("history", [])
    if not hist:
        raise ValueError("No optimization history found in training JSON.")
    coeffs = np.asarray([h["coeffs"] for h in hist], dtype=float)
    loss = np.asarray([h["loss"] for h in hist], dtype=float)
    it = np.arange(1, len(hist) + 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.semilogy(it, loss / max(d.get("initial_loss", loss[0]), 1.0e-300), marker="o", linewidth=1.2)
    ax.set_xlabel("optimization iteration")
    ax.set_ylabel(r"$J/J_0$")
    ax.set_title("VOMM adjoint optimization convergence")
    ax.grid(True, which="both", alpha=0.3)
    path = out / "vomm_optimization_convergence_optional.png"
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_run_script(outdir: str) -> Path:
    out = ensure_outdir(outdir)
    script = out / "run_full_workflow.sh"
    script.write_text("""#!/usr/bin/env bash
set -euo pipefail

# 1) Lightweight numerical validation
python src/burgers_les.py simulate --config configs/validation/moin_d2_ab2.json
python src/burgers_les.py simulate --config configs/validation/moin_d4_rk4.json
python src/postprocess.py moin \
  --d2 outputs/validation/moin_d2_ab2.npz \
  --d4 outputs/validation/moin_d4_rk4.npz \
  --outdir outputs/figures

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
python src/postprocess.py profile \
  --dns outputs/dns/dns_n8192_d4_rk4_t200.npz \
  --les outputs/les/les_n512_no_model_d4_rk4_t200.npz outputs/les/les_n512_dsm_d4_rk4_t200.npz outputs/les/les_n512_vomm_d4_rk4_t200.npz \
  --outdir outputs/figures
python src/postprocess.py kinetic \
  --dns outputs/dns/dns_n8192_d4_rk4_t200.npz \
  --les outputs/les/les_n512_no_model_d4_rk4_t200.npz outputs/les/les_n512_dsm_d4_rk4_t200.npz outputs/les/les_n512_vomm_d4_rk4_t200.npz \
  --outdir outputs/figures
python src/postprocess.py spectrum \
  --dns outputs/dns/dns_n8192_d4_rk4_t200.npz \
  --les outputs/les/les_n512_no_model_d4_rk4_t200.npz outputs/les/les_n512_dsm_d4_rk4_t200.npz outputs/les/les_n512_vomm_d4_rk4_t200.npz \
  --outdir outputs/figures --tmin 100 --tmax 200
""", encoding="utf-8")
    script.chmod(0o755)
    return script


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-process forced Burgers LES outputs")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("moin")
    p.add_argument("--d2")
    p.add_argument("--d4")
    p.add_argument("--outdir", default="figures")

    p = sub.add_parser("coeffs")
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--outdir", default="figures")

    p = sub.add_parser("profile")
    p.add_argument("--dns", required=True)
    p.add_argument("--les", nargs="+", required=True)
    p.add_argument("--outdir", default="figures")
    p.add_argument("--nx-les", type=int, default=512)

    p = sub.add_parser("kinetic")
    p.add_argument("--dns", required=True)
    p.add_argument("--les", nargs="+", required=True)
    p.add_argument("--outdir", default="figures")
    p.add_argument("--nx-les", type=int, default=512)

    p = sub.add_parser("spectrum")
    p.add_argument("--dns", required=True)
    p.add_argument("--les", nargs="+", required=True)
    p.add_argument("--outdir", default="figures")
    p.add_argument("--tmin", type=float, default=100.0)
    p.add_argument("--tmax", type=float, default=200.0)
    p.add_argument("--nx-les", type=int, default=512)

    p = sub.add_parser("vomm-training-optional")
    p.add_argument("--train-json", required=True)
    p.add_argument("--outdir", default="figures")

    p = sub.add_parser("write-run-script")
    p.add_argument("--outdir", default=".")

    args = parser.parse_args()
    if args.cmd == "moin":
        path = plot_moin(args.d2, args.d4, args.outdir)
    elif args.cmd == "coeffs":
        path = plot_coefficients(args.files, args.outdir)
    elif args.cmd == "profile":
        path = plot_profile(args.dns, args.les, args.outdir, args.nx_les)
    elif args.cmd == "kinetic":
        path = plot_resolved_k(args.dns, args.les, args.outdir, args.nx_les)
    elif args.cmd == "spectrum":
        path = plot_spectrum(args.dns, args.les, args.outdir, args.tmin, args.tmax, args.nx_les)
    elif args.cmd == "vomm-training-optional":
        path = plot_vomm_training(args.train_json, args.outdir)
    elif args.cmd == "write-run-script":
        path = write_run_script(args.outdir)
    else:
        raise RuntimeError("unreachable")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
