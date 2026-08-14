#!/usr/bin/env python3
"""
Research solver for the one-dimensional forced stochastic Burgers equation.

This revision uses a Burgers-specific entropy-conservative (quadratic-energy-conservative)
two-point flux for the nonlinear term in the second- and fourth-order physical schemes.

This file is intentionally self-contained and does not overwrite the original
pyBurgers.py / burgerslib.py files.  It adds
  * second- and fourth-order conservative finite-difference operators,
  * AB2 and RK4 time integration,
  * deterministic saving to NPZ,
  * dynamic Smagorinsky SGS,
  * a verified discrete-adjoint VOMM training workflow,
  * normalized spectral objectives, diagnostics, checkpoints, and restart support.

The VOMM implementation uses a continuous/semi-discrete adjoint of the 1D
Burgers LES ODE.  It is designed for coefficient calibration over a finite
reference window generated from filtered DNS data.  For production-quality
training, generate a reference window from the DNS using the same forcing
sequence, then optimize C1 and C2 from that window.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None


TWOPI = 2.0 * np.pi


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class SimulationConfig:
    # discretization / run length
    nx_dns: int = 8192
    nx_les: int = 512
    nt: int = 20_000
    dt: float = 1.0e-4
    visc: float = 1.0e-5
    damp: float = 1.0e-6
    beta: float = 0.75  # code convention: forcing multiplier k^(-beta/2); beta=0.75 => physical beta=-0.75
    seed: int = 1

    # numerics
    disTy: int = 4  # 0=spectral, 2=2nd-order physical, 4=4th-order physical
    time_scheme: str = "rk4"  # "ab2" or "rk4"

    # SGS model: 0 none, 1 constant Smag, 2 dynamic Smag, 3 dynamic Wong-Lilly, 5 VOMM
    sgs_model: int = 0
    vomm_coeffs: Tuple[float, float] = (0.0, 1.0)
    adm_order: int = 5
    # VOMM must use "none" during optimization so forward and adjoint are identical.
    # "hard_clip" is retained only as an explicitly labelled robustness experiment.
    vomm_backscatter_mode: str = "none"  # "none" or "hard_clip"

    # Optional numerical stabilization applied after each complete time step.
    # Keep "none" for the baseline conservative formulation.  "nyquist" removes only the
    # checkerboard mode; "spectral" is an explicitly documented high-k filter.
    numerical_stabilization: str = "none"  # "none", "nyquist", or "spectral"
    spectral_filter_cutoff: float = 0.90
    spectral_filter_strength: float = 12.0
    spectral_filter_order: int = 8

    # output
    n_info: int = 1000
    n_stat: int = 1000
    output: str = "output.npz"
    save_u: bool = True
    save_dtype: str = "float32"

    # Diagnostics/checkpoint controls.  A failed run writes a partial NPZ and a
    # restart checkpoint instead of losing the entire calculation.
    checkpoint_interval_steps: int = 100_000
    checkpoint_dir: Optional[str] = None
    restart_from: Optional[str] = None
    high_k_cutoff_fraction: float = 0.80
    abort_abs_u: float = 1.0e6

    # forcing / validation controls
    forcing: bool = True
    initial_condition: str = "zero"  # "zero" or "moin"
    moin_amplitude: float = 10.0

    # reference-window saving for VOMM training; only used by simulate
    ref_window_output: Optional[str] = None
    ref_window_start_step: Optional[int] = None
    ref_window_steps: int = 0
    ref_window_nx_les: int = 512
    ref_window_filter: str = "box"  # "box" or "spectral"

    @staticmethod
    def from_json(path: str | os.PathLike[str]) -> "SimulationConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = SimulationConfig()

        # Accept both the original namelist layout and the new flat layout.
        if "dns" in data or "les" in data or "num" in data:
            cfg.nx_dns = int(data.get("dns", {}).get("nx", cfg.nx_dns))
            cfg.nx_les = int(data.get("les", {}).get("nx", cfg.nx_les))
            cfg.sgs_model = int(data.get("les", {}).get("sgs", cfg.sgs_model))
            cfg.disTy = int(data.get("num", {}).get("disTy", cfg.disTy))
            cfg.nt = int(float(data.get("nt", cfg.nt)))
            cfg.dt = float(data.get("dt", cfg.dt))
            cfg.visc = float(data.get("visc", cfg.visc))
            cfg.damp = float(data.get("damp", cfg.damp))
            cfg.beta = float(data.get("beta", cfg.beta))
            cfg.n_info = int(data.get("nInfo", cfg.n_info))
            cfg.n_stat = int(data.get("nStat", cfg.n_stat))

        # Flat keys override the legacy ones.
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

        cfg.nx_dns = int(cfg.nx_dns)
        cfg.nx_les = int(cfg.nx_les)
        cfg.nt = int(float(cfg.nt))
        cfg.n_info = int(cfg.n_info)
        cfg.n_stat = int(cfg.n_stat)
        cfg.ref_window_steps = int(cfg.ref_window_steps)
        cfg.checkpoint_interval_steps = int(cfg.checkpoint_interval_steps)
        cfg.spectral_filter_order = int(cfg.spectral_filter_order)
        if cfg.ref_window_start_step is not None:
            cfg.ref_window_start_step = int(cfg.ref_window_start_step)
        if isinstance(cfg.vomm_coeffs, list):
            cfg.vomm_coeffs = (float(cfg.vomm_coeffs[0]), float(cfg.vomm_coeffs[1]))
        return cfg

    def to_json(self, path: str | os.PathLike[str]) -> None:
        d = asdict(self)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)


# -----------------------------------------------------------------------------
# Numerics
# -----------------------------------------------------------------------------


class Utils:
    @staticmethod
    def derivative(u: np.ndarray, dx: float, disTy: int) -> Dict[str, np.ndarray]:
        """Return dudx, du2dx and d2udx2 using conservative periodic operators."""
        u = np.asarray(u)
        n = u.size

        if disTy == 0:
            m = n // 2
            h = TWOPI / n
            fac = h / dx
            k = np.fft.fftfreq(n, d=1.0 / n)
            k[m] = 0.0
            fu = np.fft.fft(u)
            dudx = fac * np.real(np.fft.ifft(1j * k * fu))
            d2udx2 = fac**2 * np.real(np.fft.ifft(-k * k * fu))

            # 3/2-like zero-padding for the conservative nonlinear derivative.
            zero_pad = np.zeros(n, dtype=complex)
            fu_p = np.insert(fu, m, zero_pad)
            u_p = np.real(np.fft.ifft(fu_p))
            fu2_p = np.fft.fft(u_p * u_p)
            fu2 = np.concatenate((fu2_p[:m], fu2_p[n + m :]))
            du2dx = 2.0 * fac * np.real(np.fft.ifft(1j * k * fu2))
            return {"dudx": dudx, "du2dx": du2dx, "d2udx2": d2udx2}

        if disTy == 2:
            dudx = Utils.d1(u, dx, order=2)
            du2dx = Utils.burgers_ec_du2dx(u, dx, order=2)
            d2udx2 = Utils.d2(u, dx, order=2)
            return {"dudx": dudx, "du2dx": du2dx, "d2udx2": d2udx2}

        if disTy == 4:
            dudx = Utils.d1(u, dx, order=4)
            du2dx = Utils.burgers_ec_du2dx(u, dx, order=4)
            d2udx2 = Utils.d2(u, dx, order=4)
            return {"dudx": dudx, "du2dx": du2dx, "d2udx2": d2udx2}

        raise ValueError("disTy must be 0, 2, or 4")

    @staticmethod
    def burgers_ec_flux_divergence(u: np.ndarray, dx: float, radius: int) -> np.ndarray:
        """Divergence of twice the Burgers entropy-conservative flux.

        For the physical Burgers flux f(u)=u**2/2, the symmetric two-point
        entropy-conservative flux is

            f_ec(a,b) = (a**2 + a*b + b**2) / 6.

        The solver stores ``du2dx`` and later multiplies it by 1/2, so this
        routine uses H=2*f_ec and returns

            Q_r(u)_i = [H(u_i,u_{i+r}) - H(u_{i-r},u_i)] / (r*dx).

        On a periodic grid this conserves both the discrete mean and the
        quadratic invariant: sum(Q_r)=0 and dx*sum(u*Q_r)=0 (roundoff aside).
        """
        r = int(radius)
        if r <= 0:
            raise ValueError("radius must be positive")
        u = np.asarray(u)
        ur = np.roll(u, -r)
        h = (u * u + u * ur + ur * ur) / 3.0
        return (h - np.roll(h, r)) / (r * dx)

    @staticmethod
    def burgers_ec_du2dx(u: np.ndarray, dx: float, order: int) -> np.ndarray:
        """Second- or fourth-order Burgers entropy-conservative derivative of u**2."""
        q1 = Utils.burgers_ec_flux_divergence(u, dx, radius=1)
        if order == 2:
            return q1
        if order == 4:
            q2 = Utils.burgers_ec_flux_divergence(u, dx, radius=2)
            return (4.0 / 3.0) * q1 - (1.0 / 3.0) * q2
        raise ValueError("order must be 2 or 4")

    @staticmethod
    def burgers_ec_flux_divergence_vjp(
        u: np.ndarray, lam: np.ndarray, dx: float, radius: int
    ) -> np.ndarray:
        """Transpose-Jacobian action Q_r'(u)^T lam for the EC flux operator."""
        r = int(radius)
        if r <= 0:
            raise ValueError("radius must be positive")
        u = np.asarray(u)
        lam = np.asarray(lam)
        ur = np.roll(u, -r)
        # <lam,Q> = sum_i w_i H(u_i,u_{i+r}), w_i=(lam_i-lam_{i+r})/(r dx)
        w = (lam - np.roll(lam, -r)) / (r * dx)
        left = w * (2.0 * u + ur) / 3.0
        right_at_neighbor = w * (u + 2.0 * ur) / 3.0
        return left + np.roll(right_at_neighbor, r)

    @staticmethod
    def burgers_ec_du2dx_vjp(
        u: np.ndarray, lam: np.ndarray, dx: float, order: int
    ) -> np.ndarray:
        jt1 = Utils.burgers_ec_flux_divergence_vjp(u, lam, dx, radius=1)
        if order == 2:
            return jt1
        if order == 4:
            jt2 = Utils.burgers_ec_flux_divergence_vjp(u, lam, dx, radius=2)
            return (4.0 / 3.0) * jt1 - (1.0 / 3.0) * jt2
        raise ValueError("order must be 2 or 4")

    # Backward-compatible names retained for old helper scripts.
    fully_conservative_flux_divergence = burgers_ec_flux_divergence
    fully_conservative_du2dx = burgers_ec_du2dx
    fully_conservative_flux_divergence_vjp = burgers_ec_flux_divergence_vjp
    fully_conservative_du2dx_vjp = burgers_ec_du2dx_vjp

    @staticmethod
    def d1(f: np.ndarray, dx: float, order: int) -> np.ndarray:
        if order == 2:
            return (np.roll(f, -1) - np.roll(f, 1)) / (2.0 * dx)
        if order == 4:
            return (
                -np.roll(f, -2)
                + 8.0 * np.roll(f, -1)
                - 8.0 * np.roll(f, 1)
                + np.roll(f, 2)
            ) / (12.0 * dx)
        raise ValueError("order must be 2 or 4")

    @staticmethod
    def d2(f: np.ndarray, dx: float, order: int) -> np.ndarray:
        if order == 2:
            return (np.roll(f, -1) - 2.0 * f + np.roll(f, 1)) / (dx * dx)
        if order == 4:
            return (
                -np.roll(f, -2)
                + 16.0 * np.roll(f, -1)
                - 30.0 * f
                + 16.0 * np.roll(f, 1)
                - np.roll(f, 2)
            ) / (12.0 * dx * dx)
        raise ValueError("order must be 2 or 4")

    @staticmethod
    def noise(alpha: float, n: int, rng: np.random.Generator) -> np.ndarray:
        """FBM-like forcing field: F^{-1}(k^{-alpha/2} fhat)."""
        x = np.sqrt(n) * rng.standard_normal(n)
        m = n // 2
        k = np.abs(np.fft.fftfreq(n, d=1.0 / n))
        k[0] = 1.0
        fx = np.fft.fft(x)
        fx[0] = 0.0
        fx[m] = 0.0
        return np.real(np.fft.ifft(fx * k ** (-alpha / 2.0)))

    @staticmethod
    def filter_down_spectral(u: np.ndarray, ratio: int) -> np.ndarray:
        """Fourier sharp cutoff and downsample from DNS grid to LES grid."""
        if ratio == 1:
            return np.asarray(u).copy()
        n = u.size
        m = n // ratio
        l = m // 2
        fu = np.fft.fft(u)
        fuf = np.zeros(m, dtype=complex)
        fuf[:l] = fu[:l]
        fuf[l + 1 :] = fu[n - l + 1 :]
        return (1.0 / ratio) * np.real(np.fft.ifft(fuf))

    @staticmethod
    def filter_box_spectral(u: np.ndarray, ratio: int) -> np.ndarray:
        """Spectral box/sharp filter on the same grid."""
        if ratio == 1:
            return np.asarray(u).copy()
        n = u.size
        m = n // ratio
        l = m // 2
        fu = np.fft.fft(u)
        fuf = np.zeros(n, dtype=complex)
        fuf[:l] = fu[:l]
        fuf[n - l + 1 :] = fu[n - l + 1 :]
        return np.real(np.fft.ifft(fuf))

    @staticmethod
    def filter_box_ps(u: np.ndarray, ratio: int) -> np.ndarray:
        """Periodic physical-space top-hat filter with trapezoidal endpoint weights.

        For ratio=2 this gives 0.25*u[i-1] + 0.5*u[i] + 0.25*u[i+1].
        """
        ratio = int(ratio)
        u = np.asarray(u)
        if ratio <= 1:
            return u.copy()
        if ratio % 2 != 0:
            # A simple centered box with equal weights for odd ratio.
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

    @staticmethod
    def grid_filter(u: np.ndarray, ratio: int = 2, kind: str = "box") -> np.ndarray:
        if kind == "box":
            return Utils.filter_box_ps(u, ratio)
        if kind == "spectral":
            return Utils.filter_box_spectral(u, ratio)
        raise ValueError("filter kind must be 'box' or 'spectral'")

    @staticmethod
    def filter_dns_to_les(u_dns: np.ndarray, nx_les: int, kind: str = "box") -> np.ndarray:
        n = int(u_dns.size)
        if n == nx_les:
            return np.asarray(u_dns).copy()
        if n % nx_les != 0:
            raise ValueError(f"DNS size {n} is not an integer multiple of LES size {nx_les}")
        ratio = n // nx_les
        if kind == "spectral":
            return Utils.filter_down_spectral(u_dns, ratio)
        return Utils.filter_box_ps(u_dns, ratio)[::ratio].copy()

    @staticmethod
    def apply_numerical_stabilization(
        u: np.ndarray,
        mode: str,
        cutoff: float = 0.90,
        strength: float = 12.0,
        order: int = 8,
    ) -> np.ndarray:
        """Apply an explicitly selected post-step stabilization.

        ``none`` leaves the baseline conservative scheme unchanged.
        ``nyquist`` removes only the even-grid checkerboard mode.
        ``spectral`` damps only modes above ``cutoff*k_max`` with a smooth
        exponential transfer function.  The latter must be reported as an
        added numerical filter, not as an SGS closure.
        """
        mode = str(mode).lower()
        if mode == "none":
            return np.asarray(u)
        a = np.fft.fft(np.asarray(u))
        n = u.size
        if mode == "nyquist":
            if n % 2 == 0:
                a[n // 2] = 0.0
            return np.real(np.fft.ifft(a))
        if mode == "spectral":
            cutoff = float(cutoff)
            if not (0.0 < cutoff < 1.0):
                raise ValueError("spectral_filter_cutoff must be between 0 and 1")
            rho = np.abs(np.fft.fftfreq(n, d=1.0 / n)) / (n / 2.0)
            sigma = np.ones(n, dtype=float)
            mask = rho > cutoff
            eta = (rho[mask] - cutoff) / (1.0 - cutoff)
            sigma[mask] = np.exp(-float(strength) * eta ** int(order))
            a *= sigma
            if n % 2 == 0:
                a[n // 2] = 0.0
            return np.real(np.fft.ifft(a))
        raise ValueError("numerical_stabilization must be 'none', 'nyquist', or 'spectral'")

    @staticmethod
    def high_k_energy_fraction(u: np.ndarray, cutoff_fraction: float = 0.80) -> float:
        """Fraction of Fourier energy above cutoff_fraction of the Nyquist wavenumber."""
        a = np.fft.rfft(np.asarray(u))
        e = np.abs(a) ** 2
        if e.size == 0 or float(np.sum(e)) == 0.0:
            return 0.0
        rho = np.arange(e.size, dtype=float) / max(e.size - 1, 1)
        return float(np.sum(e[rho >= float(cutoff_fraction)]) / np.sum(e))


# -----------------------------------------------------------------------------
# SGS models
# -----------------------------------------------------------------------------


class SGSModels:
    @staticmethod
    def adm_deconvolution(u: np.ndarray, order: int = 5, filter_ratio: int = 2, filter_kind: str = "box") -> np.ndarray:
        """Van Cittert approximate deconvolution, sum_{r=0}^{N-1} (I-G)^r u."""
        v = np.asarray(u).copy()
        term = np.asarray(u).copy()
        for _ in range(1, int(order)):
            term = term - Utils.grid_filter(term, filter_ratio, filter_kind)
            v = v + term
        return v

    @staticmethod
    def vomm_bases(
        u: np.ndarray,
        dx: float,
        disTy: int,
        adm_order: int = 5,
        filter_ratio: int = 2,
        filter_kind: str = "box",
    ) -> Tuple[np.ndarray, np.ndarray]:
        derivs = Utils.derivative(u, dx, disTy)
        ux = derivs["dudx"]
        # T1 includes the conventional dissipative sign of the 1D Smagorinsky stress.
        t1 = -2.0 * (dx**2) * np.abs(ux) * ux

        u_star = SGSModels.adm_deconvolution(u, adm_order, filter_ratio, filter_kind)
        gu = Utils.grid_filter(u_star, filter_ratio, filter_kind)
        gu2 = Utils.grid_filter(u_star * u_star, filter_ratio, filter_kind)
        t2 = gu2 - gu * gu
        return t1, t2

    @staticmethod
    def subgrid(
        u: np.ndarray,
        dx: float,
        disTy: int,
        model: int,
        coeffs: Tuple[float, float] = (0.0, 1.0),
        adm_order: int = 5,
        filter_kind: str = "box",
        backscatter_mode: str = "none",
    ) -> Dict[str, Any]:
        n = u.size
        derivs = Utils.derivative(u, dx, disTy)
        dudx = derivs["dudx"]
        clip_fraction = 0.0

        if model == 0:
            tau = np.zeros(n)
            coeff = 0.0

        elif model == 1:
            cs2 = 0.16**2
            tau = -2.0 * cs2 * (dx**2) * np.abs(dudx) * dudx
            coeff = math.sqrt(cs2)

        elif model == 2:
            # Dynamic Smagorinsky; physical-space test filter for physical discretizations.
            if disTy == 0:
                filt = lambda a: Utils.filter_box_spectral(a, 2)
            else:
                filt = lambda a: Utils.filter_box_ps(a, 2)
            uf = filt(u)
            uuf = filt(u * u)
            l11 = uuf - uf * uf
            dudxf = filt(dudx)
            t = np.abs(dudx) * dudx
            tf = filt(t)
            m11 = -2.0 * (dx**2) * (4.0 * np.abs(dudxf) * dudxf - tf)
            den = float(np.mean(m11 * m11))
            cs2 = 0.0 if den == 0.0 else float(np.mean(l11 * m11) / den)
            cs2 = max(cs2, 0.0)
            tau = -2.0 * cs2 * (dx**2) * np.abs(dudx) * dudx
            coeff = math.sqrt(cs2)

        elif model == 3:
            # Dynamic Wong-Lilly, retained for comparisons if needed.
            if disTy == 0:
                filt = lambda a: Utils.filter_box_spectral(a, 2)
            else:
                filt = lambda a: Utils.filter_box_ps(a, 2)
            uf = filt(u)
            uuf = filt(u * u)
            l11 = uuf - uf * uf
            dudxf = filt(dudx)
            m11 = 2.0 * (dx ** (4.0 / 3.0)) * dudxf * (1.0 - 2.0 ** (4.0 / 3.0))
            den = float(np.mean(m11 * m11))
            cwl = 0.0 if den == 0.0 else float(np.mean(l11 * m11) / den)
            cwl = max(cwl, 0.0)
            tau = -2.0 * cwl * (dx ** (4.0 / 3.0)) * dudx
            coeff = cwl

        elif model == 5:
            # Variational optimal mixed-model form.  The default is deliberately
            # unclipped: this is essential for a mathematically consistent
            # forward/adjoint pair.  Hard clipping is available only as an
            # explicitly labelled robustness experiment and is never used in
            # coefficient training.
            c1, c2 = coeffs
            t1, t2 = SGSModels.vomm_bases(u, dx, disTy, adm_order=adm_order, filter_kind=filter_kind)
            tau_raw = c1 * t1 + c2 * t2
            mode = str(backscatter_mode).lower()
            if mode == "none":
                tau = tau_raw
            elif mode == "hard_clip":
                tau_diss = c1 * t1
                backscatter = tau_raw * dudx > 0.0
                tau = np.where(backscatter, tau_diss, tau_raw)
                clip_fraction = float(np.mean(backscatter))
            else:
                raise ValueError("vomm_backscatter_mode must be 'none' or 'hard_clip'")
            coeff = np.array([c1, c2], dtype=float)

        else:
            raise ValueError("Unknown SGS model. Use 0, 1, 2, 3, or 5.")

        dtaudx = Utils.derivative(tau, dx, disTy)["dudx"]
        return {"tau": tau, "dtaudx": dtaudx, "coeff": coeff, "clip_fraction": clip_fraction}


# -----------------------------------------------------------------------------
# Forward solver
# -----------------------------------------------------------------------------


def make_initial_condition(cfg: SimulationConfig) -> np.ndarray:
    x = np.arange(cfg.nx_les) * (TWOPI / cfg.nx_les)
    if cfg.initial_condition == "zero":
        return np.zeros(cfg.nx_les, dtype=float)
    if cfg.initial_condition == "moin":
        return cfg.moin_amplitude * np.sin(x)
    raise ValueError("initial_condition must be 'zero' or 'moin'")


def make_forcing_les(cfg: SimulationConfig, rng: np.random.Generator) -> np.ndarray:
    if not cfg.forcing or cfg.damp == 0.0:
        return np.zeros(cfg.nx_les)
    fbm = math.sqrt(2.0 * cfg.damp / cfg.dt) * Utils.noise(cfg.beta, cfg.nx_dns, rng)
    if cfg.nx_dns == cfg.nx_les:
        return fbm
    ratio = cfg.nx_dns // cfg.nx_les
    if cfg.disTy == 0:
        return Utils.filter_down_spectral(fbm, ratio)
    return Utils.filter_box_ps(fbm, ratio)[::ratio].copy()


def rhs(u: np.ndarray, force_les: np.ndarray, cfg: SimulationConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    dx = TWOPI / cfg.nx_les
    derivs = Utils.derivative(u, dx, cfg.disTy)
    dudx = derivs["dudx"]
    du2dx = derivs["du2dx"]
    d2udx2 = derivs["d2udx2"]
    sgs = SGSModels.subgrid(
        u,
        dx,
        cfg.disTy,
        cfg.sgs_model,
        coeffs=tuple(cfg.vomm_coeffs),
        adm_order=cfg.adm_order,
        backscatter_mode=cfg.vomm_backscatter_mode,
    )
    r = cfg.visc * d2udx2 - 0.5 * du2dx + force_les - 0.5 * sgs["dtaudx"]
    diag = {
        "dudx": dudx,
        "d2udx2": d2udx2,
        "tau": sgs["tau"],
        "coeff": sgs["coeff"],
        "clip_fraction": sgs.get("clip_fraction", 0.0),
    }
    return r, diag


def rk4_step(u: np.ndarray, force_les: np.ndarray, cfg: SimulationConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    dt = cfg.dt
    k1, diag = rhs(u, force_les, cfg)
    k2, _ = rhs(u + 0.5 * dt * k1, force_les, cfg)
    k3, _ = rhs(u + 0.5 * dt * k2, force_les, cfg)
    k4, _ = rhs(u + dt * k3, force_les, cfg)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), diag


def ab2_or_euler_step(
    u: np.ndarray,
    force_les: np.ndarray,
    cfg: SimulationConfig,
    rhs_previous: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    r, diag = rhs(u, force_les, cfg)
    if rhs_previous is None:
        u_new = u + cfg.dt * r
    else:
        u_new = u + cfg.dt * (1.5 * r - 0.5 * rhs_previous)
    return u_new, r, diag


def save_npz(path: str | os.PathLike[str], **kwargs: Any) -> None:
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **kwargs)


def _checkpoint_path(cfg: SimulationConfig, step: int, failed: bool = False) -> Path:
    out = Path(cfg.output or "burgers_les_output.npz")
    directory = Path(cfg.checkpoint_dir) if cfg.checkpoint_dir else out.parent / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    tag = "FAILED" if failed else "checkpoint"
    return directory / f"{out.stem}_{tag}_step{step:08d}.npz"


def _save_checkpoint(
    cfg: SimulationConfig,
    step: int,
    u: np.ndarray,
    rhs_prev: Optional[np.ndarray],
    rng: np.random.Generator,
    failed: bool = False,
) -> Path:
    path = _checkpoint_path(cfg, step, failed=failed)
    save_npz(
        path,
        step=np.asarray(step),
        u=np.asarray(u, dtype=np.float64),
        rhs_prev=np.asarray(rhs_prev, dtype=np.float64) if rhs_prev is not None else np.empty(0),
        rng_state_json=json.dumps(rng.bit_generator.state),
        config_json=json.dumps(asdict(cfg)),
    )
    return path


def _save_partial_result(cfg: SimulationConfig, result: Dict[str, Any], step: int) -> Path:
    out = Path(cfg.output or "burgers_les_output.npz")
    path = out.with_name(f"{out.stem}_partial_step{step:08d}.npz")
    save_npz(path, **result)
    return path


def simulate(cfg: SimulationConfig) -> Dict[str, Any]:
    if cfg.nx_dns % cfg.nx_les != 0:
        raise ValueError("nx_dns must be an integer multiple of nx_les")
    if cfg.time_scheme.lower() not in {"ab2", "rk4"}:
        raise ValueError("time_scheme must be 'ab2' or 'rk4'")
    if cfg.vomm_backscatter_mode not in {"none", "hard_clip"}:
        raise ValueError("vomm_backscatter_mode must be 'none' or 'hard_clip'")

    t0 = time.time()
    rng = np.random.default_rng(cfg.seed)
    u = make_initial_condition(cfg)
    start_step = 0
    rhs_prev: Optional[np.ndarray] = None
    if cfg.restart_from:
        restart = np.load(cfg.restart_from, allow_pickle=True)
        u = np.asarray(restart["u"], dtype=float).copy()
        start_step = int(restart["step"])
        rp = np.asarray(restart.get("rhs_prev", np.empty(0)), dtype=float)
        rhs_prev = rp.copy() if rp.size else None
        raw_state = restart.get("rng_state_json", None)
        if raw_state is not None:
            if isinstance(raw_state, np.ndarray):
                raw_state = raw_state.item()
            rng.bit_generator.state = json.loads(str(raw_state))
        print(f"[simulate] restarted from {cfg.restart_from} at step={start_step}")

    x = np.arange(cfg.nx_les) * (TWOPI / cfg.nx_les)
    dx = TWOPI / cfg.nx_les
    dtype = np.float32 if cfg.save_dtype == "float32" else np.float64

    t_hist: List[float] = []
    k_hist: List[float] = []
    coeff_hist: List[np.ndarray] = []
    diss_sgs_hist: List[float] = []
    diss_mol_hist: List[float] = []
    u_hist: List[np.ndarray] = []
    max_abs_hist: List[float] = []
    cfl_hist: List[float] = []
    high_k_hist: List[float] = []
    clip_fraction_hist: List[float] = []

    ref_u: List[np.ndarray] = []
    ref_force: List[np.ndarray] = []
    ref_t: List[float] = []
    ref_active = cfg.ref_window_output is not None and cfg.ref_window_start_step is not None and cfg.ref_window_steps > 0
    ref_start = cfg.ref_window_start_step if cfg.ref_window_start_step is not None else -1
    ref_end = ref_start + cfg.ref_window_steps

    def build_result(current_u: np.ndarray) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "x": x.astype(dtype),
            "t": np.asarray(t_hist, dtype=dtype),
            "k_resolved": np.asarray(k_hist, dtype=dtype),
            "coeff": np.vstack(coeff_hist).astype(dtype) if coeff_hist else np.empty((0, 1), dtype=dtype),
            "diss_sgs": np.asarray(diss_sgs_hist, dtype=dtype),
            "diss_mol": np.asarray(diss_mol_hist, dtype=dtype),
            "max_abs_u": np.asarray(max_abs_hist, dtype=dtype),
            "cfl": np.asarray(cfl_hist, dtype=dtype),
            "high_k_energy_fraction": np.asarray(high_k_hist, dtype=dtype),
            "vomm_clip_fraction": np.asarray(clip_fraction_hist, dtype=dtype),
            "config_json": json.dumps(asdict(cfg)),
            "u_final": np.asarray(current_u, dtype=dtype),
        }
        if cfg.save_u:
            result["u"] = np.asarray(u_hist, dtype=dtype)
        return result

    print(
        f"[simulate] nx={cfg.nx_les}, nt={cfg.nt}, dt={cfg.dt}, disTy={cfg.disTy}, "
        f"scheme={cfg.time_scheme}, SGS={cfg.sgs_model}, stabilization={cfg.numerical_stabilization}"
    )
    try:
        for step in range(start_step, cfg.nt):
            if step == start_step or ((step + 1) % cfg.n_info == 0):
                print(f"\r[simulate] step {step + 1:>8d} / {cfg.nt}", end="", flush=True)

            force_les = make_forcing_les(cfg, rng)

            if ref_active and ref_start <= step < ref_end:
                ref_u.append(Utils.filter_dns_to_les(u, cfg.ref_window_nx_les, cfg.ref_window_filter).astype(dtype))
                ref_force.append(Utils.filter_dns_to_les(force_les, cfg.ref_window_nx_les, cfg.ref_window_filter).astype(dtype))
                ref_t.append(step * cfg.dt)

            if cfg.time_scheme.lower() == "rk4":
                u, _ = rk4_step(u, force_les, cfg)
            else:
                u, rhs_prev, _ = ab2_or_euler_step(u, force_les, cfg, rhs_prev)

            u = Utils.apply_numerical_stabilization(
                u,
                cfg.numerical_stabilization,
                cfg.spectral_filter_cutoff,
                cfg.spectral_filter_strength,
                cfg.spectral_filter_order,
            )

            max_abs = float(np.max(np.abs(u))) if np.all(np.isfinite(u)) else float("inf")
            if not np.all(np.isfinite(u)) or max_abs > float(cfg.abort_abs_u):
                checkpoint = _save_checkpoint(cfg, step + 1, u, rhs_prev, rng, failed=True)
                partial = _save_partial_result(cfg, build_result(u), step + 1)
                raise FloatingPointError(
                    f"Solution became non-finite/unbounded at step {step + 1}, t={(step + 1)*cfg.dt:.6g}. "
                    f"Saved {checkpoint} and {partial}."
                )

            if ref_active and step == ref_end - 1:
                ref_u.append(Utils.filter_dns_to_les(u, cfg.ref_window_nx_les, cfg.ref_window_filter).astype(dtype))
                ref_t.append((step + 1) * cfg.dt)

            if cfg.checkpoint_interval_steps > 0 and (step + 1) % cfg.checkpoint_interval_steps == 0:
                path = _save_checkpoint(cfg, step + 1, u, rhs_prev, rng, failed=False)
                print(f"\n[simulate] checkpoint: {path}")

            if (step + 1) % cfg.n_stat == 0:
                derivs = Utils.derivative(u, dx, cfg.disTy)
                dudx = derivs["dudx"]
                sgs = SGSModels.subgrid(
                    u,
                    dx,
                    cfg.disTy,
                    cfg.sgs_model,
                    coeffs=tuple(cfg.vomm_coeffs),
                    adm_order=cfg.adm_order,
                    backscatter_mode=cfg.vomm_backscatter_mode,
                )
                tau = sgs["tau"]
                coeff = np.atleast_1d(sgs["coeff"]).astype(float)
                t_hist.append((step + 1) * cfg.dt)
                k_hist.append(0.5 * float(np.mean(u * u)))
                coeff_hist.append(coeff)
                diss_sgs_hist.append(float(np.mean(-tau * dudx)))
                diss_mol_hist.append(float(np.mean(cfg.visc * dudx * dudx)))
                max_abs_hist.append(max_abs)
                cfl_hist.append(max_abs * cfg.dt / dx)
                high_k_hist.append(Utils.high_k_energy_fraction(u, cfg.high_k_cutoff_fraction))
                clip_fraction_hist.append(float(sgs.get("clip_fraction", 0.0)))
                if cfg.save_u:
                    u_hist.append(u.astype(dtype))

    except Exception:
        print()
        raise

    print(f"\n[simulate] completed in {time.time() - t0:.2f} s")
    result = build_result(u)

    if cfg.output:
        save_npz(cfg.output, **result)
        print(f"[simulate] wrote {cfg.output}")

    if ref_active:
        ref = {
            "t": np.asarray(ref_t, dtype=dtype),
            "u_ref": np.asarray(ref_u, dtype=dtype),
            "force_les": np.asarray(ref_force, dtype=dtype),
            "dt": np.asarray(cfg.dt),
            "dx": np.asarray(TWOPI / cfg.ref_window_nx_les),
            "nx_les": np.asarray(cfg.ref_window_nx_les),
            "config_json": json.dumps(asdict(cfg)),
        }
        save_npz(cfg.ref_window_output, **ref)
        print(f"[simulate] wrote reference window {cfg.ref_window_output}")

    return result


# -----------------------------------------------------------------------------
# VOMM adjoint training
# -----------------------------------------------------------------------------


@dataclass
class VOMMTrainConfig:
    ref: str
    output: str = "vomm_coeffs.json"
    disTy: int = 4
    time_scheme: str = "rk4"
    visc: float = 1.0e-5
    adm_order: int = 5
    initial_c1: float = 0.0256
    initial_c2: float = 0.10
    maxiter: int = 50
    cost_stride: int = 10
    # Optional independent short windows [start_step, number_of_steps].
    # Each window is initialized from filtered DNS, avoiding a single long
    # chaotic trajectory while sampling different stationary times.
    segments: Optional[List[Tuple[int, int]]] = None
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.0, 0.25), (-2.0, 2.0))
    normalize_cost: bool = True
    normalization_eps: float = 1.0e-30
    gradient_method: str = "discrete_adjoint"  # "discrete_adjoint" or "finite_difference"
    fd_step: float = 1.0e-5
    gradient_check: bool = True
    gradient_check_tol: float = 5.0e-4
    fallback_to_finite_difference: bool = True

    @staticmethod
    def from_json(path: str | os.PathLike[str]) -> "VOMMTrainConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = VOMMTrainConfig(ref=data["ref"])
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        if isinstance(cfg.bounds, list):
            cfg.bounds = tuple(tuple(float(x) for x in pair) for pair in cfg.bounds)  # type: ignore
        cfg.disTy = int(cfg.disTy)
        cfg.adm_order = int(cfg.adm_order)
        cfg.maxiter = int(cfg.maxiter)
        cfg.cost_stride = int(cfg.cost_stride)
        if cfg.segments is not None:
            cfg.segments = [(int(a), int(b)) for a, b in cfg.segments]
        return cfg


def dissipation_spectrum(u: np.ndarray, visc: float, dx: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = u.size
    a = np.fft.rfft(u)
    k = np.fft.rfftfreq(n, d=dx) * TWOPI
    weights = np.ones_like(k)
    if weights.size > 2:
        weights[1:-1] = 2.0
    q = visc * k * k * weights * 0.5 / (n * n)
    d = q * np.abs(a) ** 2
    return k, d, q


def spectrum_loss_and_grad(
    u: np.ndarray,
    d_ref: np.ndarray,
    visc: float,
    dx: float,
    normalize: bool = False,
    eps: float = 1.0e-30,
) -> Tuple[float, np.ndarray, np.ndarray]:
    n = u.size
    a = np.fft.rfft(u)
    _, d, q = dissipation_spectrum(u, visc, dx)
    e = d - d_ref
    denom = float(np.sum(d_ref * d_ref) + eps) if normalize else 1.0
    loss = float(np.sum(e * e) / denom)

    # Exact gradient of the spectral discrepancy with respect to the real state.
    z = 2.0 * e * q * a
    if z.size > 0:
        z[0] = 4.0 * e[0] * q[0] * a[0]
    if z.size > 1:
        z[-1] = 4.0 * e[-1] * q[-1] * a[-1]
    grad = (n * np.fft.irfft(z, n)) / denom
    return loss, grad, d


def cost_reference_spectra(u_ref: np.ndarray, visc: float, dx: float) -> np.ndarray:
    return np.asarray([dissipation_spectrum(u, visc, dx)[1] for u in u_ref])


class VOMMObjective:
    def __init__(self, cfg: VOMMTrainConfig):
        self.cfg = cfg
        ref_npz = np.load(cfg.ref, allow_pickle=True)
        self.u_ref = np.asarray(ref_npz["u_ref"], dtype=float)
        self.forces = np.asarray(ref_npz["force_les"], dtype=float)
        self.dt = float(ref_npz["dt"])
        self.dx = float(ref_npz["dx"])
        self.nx = int(ref_npz["nx_les"])
        if self.u_ref.shape[0] != self.forces.shape[0] + 1:
            raise ValueError("Reference window must contain len(force_les)+1 u_ref states")
        self.d_ref = cost_reference_spectra(self.u_ref, cfg.visc, self.dx)
        if cfg.segments:
            self.segments = [(int(a), int(b)) for a, b in cfg.segments]
        else:
            self.segments = [(0, self.forces.shape[0])]
        for start, length in self.segments:
            if start < 0 or length <= 0 or start + length > self.forces.shape[0]:
                raise ValueError(
                    f"Invalid training segment [{start}, {length}]; reference has {self.forces.shape[0]} steps"
                )
        self.last: Dict[str, Any] = {}

    def _sim_cfg(self, coeffs: np.ndarray) -> SimulationConfig:
        return SimulationConfig(
            nx_dns=self.nx,
            nx_les=self.nx,
            nt=max(length for _, length in self.segments),
            dt=self.dt,
            visc=self.cfg.visc,
            damp=0.0,
            disTy=self.cfg.disTy,
            time_scheme=self.cfg.time_scheme,
            sgs_model=5,
            vomm_coeffs=(float(coeffs[0]), float(coeffs[1])),
            adm_order=self.cfg.adm_order,
            vomm_backscatter_mode="none",
            numerical_stabilization="none",
            forcing=False,
            n_info=10**12,
            n_stat=10**12,
            output="",
        )

    def forward_segment(self, coeffs: np.ndarray, start: int, length: int) -> np.ndarray:
        sim_cfg = self._sim_cfg(coeffs)
        u_hist = np.empty((length + 1, self.nx), dtype=float)
        u = self.u_ref[start].astype(float).copy()
        u_hist[0] = u
        rhs_prev: Optional[np.ndarray] = None
        for local in range(length):
            f = self.forces[start + local]
            if self.cfg.time_scheme.lower() == "rk4":
                u, _ = rk4_step(u, f, sim_cfg)
            else:
                u, rhs_prev, _ = ab2_or_euler_step(u, f, sim_cfg, rhs_prev)
            if not np.all(np.isfinite(u)):
                raise FloatingPointError(
                    f"Non-finite VOMM training trajectory in segment start={start}, local step={local+1}"
                )
            u_hist[local + 1] = u
        return u_hist

    def _loss_and_state_gradient(
        self, u_hist: np.ndarray, global_start: int
    ) -> Tuple[float, np.ndarray]:
        loss = 0.0
        lgrad = np.zeros_like(u_hist)
        obs = np.arange(0, u_hist.shape[0], max(1, int(self.cfg.cost_stride)), dtype=int)
        if obs[-1] != u_hist.shape[0] - 1:
            obs = np.append(obs, u_hist.shape[0] - 1)
        nobs = float(obs.size)
        for local in obs:
            global_index = global_start + int(local)
            l, g, _ = spectrum_loss_and_grad(
                u_hist[local],
                self.d_ref[global_index],
                self.cfg.visc,
                self.dx,
                normalize=self.cfg.normalize_cost,
                eps=self.cfg.normalization_eps,
            )
            loss += l / nobs
            lgrad[local] += g / nobs
        return float(loss), lgrad

    def loss_only(self, coeffs: np.ndarray) -> float:
        coeffs = np.asarray(coeffs, dtype=float)
        total = 0.0
        for start, length in self.segments:
            u_hist = self.forward_segment(coeffs, start, length)
            total += self._loss_and_state_gradient(u_hist, start)[0]
        return float(total / len(self.segments))

    def finite_difference_gradient(self, coeffs: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(coeffs, dtype=float)
        grad = np.zeros_like(coeffs)
        for i in range(coeffs.size):
            h = float(self.cfg.fd_step) * max(1.0, abs(float(coeffs[i])))
            xp = coeffs.copy(); xm = coeffs.copy()
            xp[i] += h; xm[i] -= h
            grad[i] = (self.loss_only(xp) - self.loss_only(xm)) / (2.0 * h)
        return grad

    def _rk4_reverse_step(
        self,
        u: np.ndarray,
        force: np.ndarray,
        adj_next: np.ndarray,
        sim_cfg: SimulationConfig,
    ) -> Tuple[np.ndarray, np.ndarray]:
        dt = self.dt
        k1, _ = rhs(u, force, sim_cfg)
        u2 = u + 0.5 * dt * k1
        k2, _ = rhs(u2, force, sim_cfg)
        u3 = u + 0.5 * dt * k2
        k3, _ = rhs(u3, force, sim_cfg)
        u4 = u + dt * k3
        rhs(u4, force, sim_cfg)  # mirror the k4 stage of the forward graph

        adj_u = adj_next.copy()
        adj_k1 = (dt / 6.0) * adj_next
        adj_k2 = (dt / 3.0) * adj_next
        adj_k3 = (dt / 3.0) * adj_next
        adj_k4 = (dt / 6.0) * adj_next
        grad = np.zeros(2, dtype=float)

        grad += vomm_param_gradient(u4, adj_k4, self.dx, sim_cfg)
        adj_u4 = vomm_jacobian_transpose_action(u4, adj_k4, self.dx, sim_cfg, stabilize=False)
        adj_u += adj_u4
        adj_k3 += dt * adj_u4

        grad += vomm_param_gradient(u3, adj_k3, self.dx, sim_cfg)
        adj_u3 = vomm_jacobian_transpose_action(u3, adj_k3, self.dx, sim_cfg, stabilize=False)
        adj_u += adj_u3
        adj_k2 += 0.5 * dt * adj_u3

        grad += vomm_param_gradient(u2, adj_k2, self.dx, sim_cfg)
        adj_u2 = vomm_jacobian_transpose_action(u2, adj_k2, self.dx, sim_cfg, stabilize=False)
        adj_u += adj_u2
        adj_k1 += 0.5 * dt * adj_u2

        grad += vomm_param_gradient(u, adj_k1, self.dx, sim_cfg)
        adj_u += vomm_jacobian_transpose_action(u, adj_k1, self.dx, sim_cfg, stabilize=False)
        return adj_u, grad

    def discrete_adjoint_gradient(
        self,
        coeffs: np.ndarray,
        u_hist: np.ndarray,
        lgrad: np.ndarray,
        forces: np.ndarray,
    ) -> np.ndarray:
        sim_cfg = self._sim_cfg(coeffs)
        nt = forces.shape[0]
        grad = np.zeros(2, dtype=float)
        scheme = self.cfg.time_scheme.lower()
        if scheme == "rk4":
            adj = lgrad[-1].copy()
            for n in range(nt - 1, -1, -1):
                adj, gstep = self._rk4_reverse_step(u_hist[n], forces[n], adj, sim_cfg)
                grad += gstep
                adj += lgrad[n]
            return grad

        # Exact reverse of Euler startup followed by AB2.
        adj_states = lgrad.copy()
        for n in range(nt - 1, -1, -1):
            adj_out = adj_states[n + 1].copy()
            adj_states[n] += adj_out
            a = 1.0 if n == 0 else 1.5
            wa = (a * self.dt) * adj_out
            grad += vomm_param_gradient(u_hist[n], wa, self.dx, sim_cfg)
            adj_states[n] += vomm_jacobian_transpose_action(u_hist[n], wa, self.dx, sim_cfg, stabilize=False)
            if n >= 1:
                wb = (-0.5 * self.dt) * adj_out
                grad += vomm_param_gradient(u_hist[n - 1], wb, self.dx, sim_cfg)
                adj_states[n - 1] += vomm_jacobian_transpose_action(
                    u_hist[n - 1], wb, self.dx, sim_cfg, stabilize=False
                )
        return grad

    def value_and_grad(self, coeffs: np.ndarray, method: Optional[str] = None) -> Tuple[float, np.ndarray]:
        coeffs = np.asarray(coeffs, dtype=float)
        method = str(method or self.cfg.gradient_method).lower()
        if method == "finite_difference":
            loss = self.loss_only(coeffs)
            grad = self.finite_difference_gradient(coeffs)
            self.last = {"loss": loss, "coeffs": coeffs.copy(), "grad": grad.copy()}
            return float(loss), grad
        if method != "discrete_adjoint":
            raise ValueError("gradient_method must be 'discrete_adjoint' or 'finite_difference'")

        total_loss = 0.0
        total_grad = np.zeros(2, dtype=float)
        for start, length in self.segments:
            u_hist = self.forward_segment(coeffs, start, length)
            loss, lgrad = self._loss_and_state_gradient(u_hist, start)
            forces = self.forces[start : start + length]
            total_loss += loss
            total_grad += self.discrete_adjoint_gradient(coeffs, u_hist, lgrad, forces)
        total_loss /= len(self.segments)
        total_grad /= len(self.segments)
        self.last = {"loss": total_loss, "coeffs": coeffs.copy(), "grad": total_grad.copy()}
        return float(total_loss), total_grad


def vomm_param_gradient(u: np.ndarray, lam: np.ndarray, dx: float, cfg: SimulationConfig) -> np.ndarray:
    t1, t2 = SGSModels.vomm_bases(u, dx, cfg.disTy, cfg.adm_order)
    dlam = Utils.derivative(lam, dx, cfg.disTy)["dudx"]
    # dF/dC_i = -0.5 D(T_i); using D^T=-D gives +0.5 <D lam, T_i>.
    g1 = 0.5 * float(np.mean(dlam * t1)) * u.size
    g2 = 0.5 * float(np.mean(dlam * t2)) * u.size
    return np.array([g1, g2], dtype=float)


def vomm_jacobian_transpose_action(
    u: np.ndarray,
    lam: np.ndarray,
    dx: float,
    cfg: SimulationConfig,
    stabilize: bool = True,
) -> np.ndarray:
    """Apply the transpose of the semi-discrete RHS Jacobian to lam.

    Forward RHS:
        F = nu D2 u - 0.5 Q_EC(u) - 0.5 D(C1*T1 + C2*T2) + forcing
    where Q_EC is the second- or fourth-order Burgers entropy-conservative flux operator.
    with T1=-2*dx^2*|u_x|u_x and T2=G(v^2)-(Gv)^2, v=AD_N(u).
    """
    c1, c2 = cfg.vomm_coeffs
    order = cfg.disTy
    if order == 0:
        # The VOMM training workflow is intended for physical schemes.
        raise ValueError("Adjoint VOMM training currently supports physical disTy=2 or 4")

    ux = Utils.d1(u, dx, order)
    dlam = Utils.d1(lam, dx, order)
    d2lam = Utils.d2(lam, dx, order)

    # Linear diffusion and the exact transpose of the Burgers entropy-conservative
    # nonlinear operator -0.5*Q(u).
    jt = cfg.visc * d2lam - 0.5 * Utils.burgers_ec_du2dx_vjp(
        u, lam, dx, order
    )

    # T1 contribution: L^T lam = 2*C1*dx^2*D(|u_x| D lam)
    if c1 != 0.0:
        jt += 2.0 * c1 * (dx**2) * Utils.d1(np.abs(ux) * dlam, dx, order)

    # T2/ADM contribution.
    if c2 != 0.0:
        v = SGSModels.adm_deconvolution(u, cfg.adm_order)
        gv = Utils.grid_filter(v, 2, "box")
        gdlam = Utils.grid_filter(dlam, 2, "box")
        g_gv_dlam = Utils.grid_filter(gv * dlam, 2, "box")
        inner = v * gdlam - g_gv_dlam
        # A^T = A for symmetric box filters and Van Cittert AD.
        jt += c2 * SGSModels.adm_deconvolution(inner, cfg.adm_order)

    if stabilize:
        # 1D minimal stabilization analogue of the stabilized adjoint equations:
        # when u_x < 0, the quadratic adjoint production is clipped by adding u_x*lambda.
        jt += np.where(ux < 0.0, ux * lam, 0.0)

    return jt


def run_vomm_training(cfg: VOMMTrainConfig) -> Dict[str, Any]:
    if minimize is None:
        raise RuntimeError("scipy.optimize.minimize is required for VOMM training")
    obj = VOMMObjective(cfg)
    x0 = np.array([cfg.initial_c1, cfg.initial_c2], dtype=float)
    method_used = str(cfg.gradient_method).lower()
    gradient_check_record: Dict[str, Any] = {}

    if cfg.gradient_check:
        loss, adj = obj.value_and_grad(x0, method="discrete_adjoint")
        fd = obj.finite_difference_gradient(x0)
        rel = float(np.linalg.norm(adj - fd) / max(np.linalg.norm(fd), 1.0e-14))
        gradient_check_record = {
            "loss": float(loss),
            "discrete_adjoint": adj.tolist(),
            "finite_difference": fd.tolist(),
            "relative_error": rel,
            "tolerance": float(cfg.gradient_check_tol),
        }
        print(f"[train-vomm] gradient check: adjoint={adj}, finite-diff={fd}, relerr={rel:.3e}")
        if method_used == "discrete_adjoint" and rel > float(cfg.gradient_check_tol):
            if cfg.fallback_to_finite_difference:
                print("[train-vomm] WARNING: gradient check failed; falling back to finite differences")
                method_used = "finite_difference"
            else:
                raise RuntimeError(
                    f"Discrete-adjoint gradient check failed: relerr={rel:.3e} > {cfg.gradient_check_tol:.3e}"
                )

    history: List[Dict[str, Any]] = []
    initial_loss = obj.loss_only(x0)
    print(f"[train-vomm] initial coeffs={x0}, normalized loss={initial_loss:.6e}, method={method_used}")

    def fun(x: np.ndarray) -> Tuple[float, np.ndarray]:
        l, g = obj.value_and_grad(x, method=method_used)
        if not np.all(np.isfinite(g)) or not np.isfinite(l):
            raise FloatingPointError("Non-finite VOMM objective/gradient")
        return l, g

    def callback(xk: np.ndarray) -> None:
        l = float(obj.last.get("loss", np.nan))
        g = np.asarray(obj.last.get("grad", np.array([np.nan, np.nan])))
        history.append({"coeffs": xk.tolist(), "loss": l, "grad": g.tolist()})
        rel = l / initial_loss if initial_loss != 0.0 else np.nan
        print(f"[train-vomm] iter={len(history):03d}, coeffs={xk}, loss={l:.6e}, rel={rel:.4e}")

    res = minimize(
        fun=fun,
        x0=x0,
        method="L-BFGS-B",
        jac=True,
        bounds=cfg.bounds,
        options={"maxiter": int(cfg.maxiter), "ftol": 1.0e-12, "gtol": 1.0e-8, "maxls": 30},
        callback=callback,
    )

    output = {
        "success": bool(res.success),
        "message": str(res.message),
        "gradient_method_requested": cfg.gradient_method,
        "gradient_method_used": method_used,
        "gradient_check": gradient_check_record,
        "initial_coeffs": x0.tolist(),
        "optimal_coeffs": np.asarray(res.x, dtype=float).tolist(),
        "initial_loss": float(initial_loss),
        "final_loss": float(res.fun),
        "relative_final_loss": float(res.fun / initial_loss) if initial_loss != 0.0 else None,
        "history": history,
        "train_config": asdict(cfg),
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"[train-vomm] wrote {cfg.output}")
    return output


def apply_vomm_coeffs(train_json: str, config: str, out: str, result_output: Optional[str] = None) -> None:
    with open(train_json, "r", encoding="utf-8") as f:
        trained = json.load(f)
    with open(config, "r", encoding="utf-8") as f:
        data = json.load(f)
    coeffs = trained.get("optimal_coeffs")
    if coeffs is None or len(coeffs) != 2:
        raise ValueError("Training JSON does not contain two optimal_coeffs")
    data["vomm_coeffs"] = [float(coeffs[0]), float(coeffs[1])]
    data["vomm_backscatter_mode"] = "none"
    if result_output is not None:
        data["output"] = result_output
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[apply-vomm-coeffs] wrote {out} with coeffs={data['vomm_coeffs']}")


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------


def run_numerics_self_test() -> None:
    print("[self-test] derivative accuracy on sin/cos")
    for n in [32, 64, 128]:
        x = np.arange(n) * TWOPI / n
        f = np.sin(3 * x) + 0.25 * np.cos(5 * x)
        f1 = 3 * np.cos(3 * x) - 1.25 * np.sin(5 * x)
        f2 = -9 * np.sin(3 * x) - 6.25 * np.cos(5 * x)
        for order in [2, 4]:
            dx = TWOPI / n
            e1 = np.linalg.norm(Utils.d1(f, dx, order) - f1) / np.linalg.norm(f1)
            e2 = np.linalg.norm(Utils.d2(f, dx, order) - f2) / np.linalg.norm(f2)
            print(f"  n={n:4d}, D{order}: relerr d1={e1:.3e}, d2={e2:.3e}")

    print("[self-test] Burgers EC nonlinear derivative accuracy")
    previous = {2: None, 4: None}
    for nfc in [32, 64, 128, 256]:
        xfc = np.arange(nfc) * TWOPI / nfc
        ufc = 0.7 * np.sin(2.0 * xfc) + 0.2 * np.cos(3.0 * xfc)
        exact = 2.0 * ufc * (1.4 * np.cos(2.0 * xfc) - 0.6 * np.sin(3.0 * xfc))
        for order in [2, 4]:
            approx = Utils.burgers_ec_du2dx(ufc, TWOPI / nfc, order)
            err = np.linalg.norm(approx - exact) / np.linalg.norm(exact)
            rate = ""
            if previous[order] is not None:
                rate = f", rate={np.log(previous[order] / err) / np.log(2.0):.2f}"
            print(f"  n={nfc:4d}, EC-D{order}: relerr={err:.3e}{rate}")
            previous[order] = err

    print("[self-test] Burgers EC nonlinear VJP and conservation checks")
    rng_fc = np.random.default_rng(123)
    nfc = 48
    dxfc = TWOPI / nfc
    ufc = 0.2 * rng_fc.standard_normal(nfc)
    lamfc = rng_fc.standard_normal(nfc)
    deltafc = rng_fc.standard_normal(nfc)
    epsfc = 1.0e-7
    for order in [2, 4]:
        qp = Utils.burgers_ec_du2dx(ufc + epsfc * deltafc, dxfc, order)
        qm = Utils.burgers_ec_du2dx(ufc - epsfc * deltafc, dxfc, order)
        jdelta = (qp - qm) / (2.0 * epsfc)
        jtlam = Utils.burgers_ec_du2dx_vjp(ufc, lamfc, dxfc, order)
        lhsfc = float(np.dot(lamfc, jdelta))
        rhsfc = float(np.dot(deltafc, jtlam))
        print(f"  EC-D{order}: lhs={lhsfc:.6e}, rhs={rhsfc:.6e}, absdiff={abs(lhsfc-rhsfc):.3e}")
        qfc = Utils.burgers_ec_du2dx(ufc, dxfc, order)
        mass_res = float(dxfc * np.sum(qfc))
        energy_res = float(dxfc * np.dot(ufc, qfc))
        print(f"         mean-residual={mass_res:.3e}, quadratic-energy-residual={energy_res:.3e}")

    print("[self-test] spectrum gradient finite-difference check")
    rng = np.random.default_rng(0)
    n = 32
    dx = TWOPI / n
    u = rng.standard_normal(n)
    d_ref = np.ones(n // 2 + 1) * 1.0e-4
    loss, grad, _ = spectrum_loss_and_grad(u, d_ref, 1.0e-5, dx)
    fd = np.zeros(n)
    eps = 1.0e-6
    for j in range(n):
        up = u.copy(); um = u.copy()
        up[j] += eps; um[j] -= eps
        lp = spectrum_loss_and_grad(up, d_ref, 1.0e-5, dx)[0]
        lm = spectrum_loss_and_grad(um, d_ref, 1.0e-5, dx)[0]
        fd[j] = (lp - lm) / (2 * eps)
    rel = np.linalg.norm(grad - fd) / max(np.linalg.norm(fd), 1.0e-30)
    print(f"  spectrum gradient relerr={rel:.3e}")

    print("[self-test] VOMM J^T dot-product check")
    cfg = SimulationConfig(
        nx_dns=n, nx_les=n, disTy=4, sgs_model=5, vomm_coeffs=(0.02, 1.0),
        vomm_backscatter_mode="none", forcing=False
    )
    u = rng.standard_normal(n) * 0.1
    lam = rng.standard_normal(n)
    delta = rng.standard_normal(n)
    f0, _ = rhs(u, np.zeros(n), cfg)
    eps = 1.0e-6
    fp, _ = rhs(u + eps * delta, np.zeros(n), cfg)
    fm, _ = rhs(u - eps * delta, np.zeros(n), cfg)
    j_delta = (fp - fm) / (2.0 * eps)
    jt_lam = vomm_jacobian_transpose_action(u, lam, TWOPI / n, cfg, stabilize=False)
    lhs = float(np.dot(lam, j_delta))
    rhs_dot = float(np.dot(delta, jt_lam))
    print(f"  <lam,J delta>={lhs:.6e}, <J^T lam,delta>={rhs_dot:.6e}, absdiff={abs(lhs-rhs_dot):.3e}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="1D stochastic Burgers LES solver with adjoint-based VOMM support")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sim = sub.add_parser("simulate", help="run a DNS/LES simulation from a JSON config")
    p_sim.add_argument("--config", required=True)

    p_train = sub.add_parser("train-vomm", help="train VOMM coefficients from a reference-window NPZ")
    p_train.add_argument("--config", required=True)

    p_apply = sub.add_parser("apply-vomm-coeffs", help="insert trained coefficients into an LES config")
    p_apply.add_argument("--train-json", required=True)
    p_apply.add_argument("--config", required=True)
    p_apply.add_argument("--out", required=True)
    p_apply.add_argument("--result-output", default=None)

    p_test = sub.add_parser("self-test", help="run numerical self-tests")

    p_template = sub.add_parser("write-templates", help="write example JSON configs")
    p_template.add_argument("--dir", default="configs_project2")

    args = parser.parse_args(argv)

    if args.cmd == "simulate":
        cfg = SimulationConfig.from_json(args.config)
        simulate(cfg)
        return 0

    if args.cmd == "train-vomm":
        cfg = VOMMTrainConfig.from_json(args.config)
        run_vomm_training(cfg)
        return 0

    if args.cmd == "apply-vomm-coeffs":
        apply_vomm_coeffs(args.train_json, args.config, args.out, args.result_output)
        return 0

    if args.cmd == "self-test":
        run_numerics_self_test()
        return 0

    if args.cmd == "write-templates":
        write_template_configs(args.dir)
        return 0

    raise RuntimeError("unreachable")


def write_template_configs(directory: str | os.PathLike[str]) -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)

    validation = SimulationConfig(
        nx_dns=32,
        nx_les=32,
        nt=200,
        dt=0.005,
        visc=1.0,
        damp=0.0,
        forcing=False,
        initial_condition="moin",
        disTy=4,
        time_scheme="rk4",
        sgs_model=0,
        n_info=20,
        n_stat=20,
        output=str(d / "validation_moin_D4_RK4.npz"),
    )
    validation.to_json(d / "validation_moin_D4_RK4.json")

    dns_short_ref = SimulationConfig(
        nx_dns=8192,
        nx_les=8192,
        nt=20_000,
        dt=1.0e-4,
        visc=1.0e-5,
        damp=1.0e-6,
        beta=0.75,
        disTy=4,
        time_scheme="rk4",
        sgs_model=0,
        n_info=1000,
        n_stat=1000,
        output=str(d / "dns_D4_RK4_short.npz"),
        ref_window_output=str(d / "ref_window_short_les512.npz"),
        ref_window_start_step=10_000,
        ref_window_steps=1000,
        ref_window_nx_les=512,
    )
    dns_short_ref.to_json(d / "dns_D4_RK4_short_with_ref_window.json")

    dns_full = SimulationConfig(
        nx_dns=8192,
        nx_les=8192,
        nt=2_000_000,
        dt=1.0e-4,
        visc=1.0e-5,
        damp=1.0e-6,
        beta=0.75,
        disTy=4,
        time_scheme="rk4",
        sgs_model=0,
        n_info=10_000,
        n_stat=1000,
        output=str(d / "dns_N8192_D4_RK4_T200.npz"),
        ref_window_output=str(d / "ref_window_T100_T101_les512.npz"),
        ref_window_start_step=1_000_000,
        ref_window_steps=10_000,
        ref_window_nx_les=512,
    )
    dns_full.to_json(d / "dns_N8192_D4_RK4_T200_with_ref_window.json")

    les_dsm = SimulationConfig(
        nx_dns=8192,
        nx_les=512,
        nt=2_000_000,
        dt=1.0e-4,
        visc=1.0e-5,
        damp=1.0e-6,
        beta=0.75,
        disTy=4,
        time_scheme="rk4",
        sgs_model=2,
        n_info=10_000,
        n_stat=1000,
        output=str(d / "les_N512_DSM_D4_RK4_T200.npz"),
    )
    les_dsm.to_json(d / "les_N512_DSM_D4_RK4_T200.json")

    les_nomodel = SimulationConfig(**{**asdict(les_dsm), "sgs_model": 0, "output": str(d / "les_N512_NOMODEL_D4_RK4_T200.npz")})
    les_nomodel.to_json(d / "les_N512_NOMODEL_D4_RK4_T200.json")

    les_vomm = SimulationConfig(**{**asdict(les_dsm), "sgs_model": 5, "vomm_coeffs": (0.0, 1.0), "output": str(d / "les_N512_VOMM_D4_RK4_T200.npz")})
    les_vomm.to_json(d / "les_N512_VOMM_D4_RK4_T200.json")

    train = VOMMTrainConfig(
        ref=str(d / "ref_window_T100_T101_les512.npz"),
        output=str(d / "vomm_coeffs_T100_T101.json"),
        maxiter=30,
        cost_stride=10,
        gradient_check=True,
        gradient_method="discrete_adjoint",
        normalize_cost=True,
    )
    with open(d / "train_vomm_T100_T101.json", "w", encoding="utf-8") as f:
        json.dump(asdict(train), f, indent=2)

    print(f"[templates] wrote configs to {d}")


if __name__ == "__main__":
    raise SystemExit(main())
