"""Analysis utilities for the spiking CPG Arnold-tongue sweep."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from cpg_sim import run_sim, split_into_bursts

# Default classification thresholds. They are intentionally conservative.
FREQ_TOL = 0.025          # relative 1:1 frequency-match tolerance
R_TOL = 0.90              # 1:1 phase-locking value threshold
SLOPE_TOL = 0.20          # rad/s, residual phase-drift tolerance
MAX_CV = 0.35             # cycle-period CV tolerance for clean oscillation
MIN_ALT_FRAC = 0.70       # within-segment alternation threshold
MIN_BALANCE = 0.35        # lower ratio of A/B burst counts
MIN_CYCLES = 4


def _filter_spikes(spikes: np.ndarray, t_start: float) -> np.ndarray:
    spikes = np.asarray(spikes, dtype=float)
    return spikes[spikes >= t_start]


def _burst_onsets(spikes: np.ndarray, isi_gap: float) -> np.ndarray:
    return np.array([b[0] for b in split_into_bursts(spikes, isi_gap)], dtype=float)


def _cv(x: np.ndarray) -> float:
    if len(x) < 2 or np.mean(x) <= 0:
        return np.nan
    return float(np.std(x) / np.mean(x))


def _alternation_fraction(ref: np.ndarray, other: np.ndarray) -> float:
    """Fraction of ref intervals containing at least one other burst."""
    if len(ref) < 2 or len(other) == 0:
        return 0.0
    count = 0
    for i in range(len(ref) - 1):
        if np.any((other > ref[i]) & (other < ref[i + 1])):
            count += 1
    return count / (len(ref) - 1)


def _mean_spikes_per_burst(spikes: np.ndarray, isi_gap: float) -> float:
    bursts = split_into_bursts(spikes, isi_gap)
    if not bursts:
        return 0.0
    return float(np.mean([len(b) for b in bursts]))


def analyze_segment(
    spkA: np.ndarray,
    spkB: np.ndarray,
    t_start: float = 0.0,
    isi_gap: float = 0.15,
    min_cycles: int = MIN_CYCLES,
    max_cv: float = MAX_CV,
    min_alt_frac: float = MIN_ALT_FRAC,
    min_balance: float = MIN_BALANCE,
) -> Dict[str, object]:
    """Classify one half-center oscillator.

    Returns a dict containing status, frequency, reference phase onsets,
    burst counts, alternation statistics, and burst regularity.

    Status labels:
        osc         clean alternating oscillation
        silent      too few spikes to characterize
        asymmetric  one neuron dominates or the two neurons do not both burst
        tonic       sustained firing without clean alternation
        irregular   bursts exist but period CV is too high
    """
    sA = _filter_spikes(spkA, t_start)
    sB = _filter_spikes(spkB, t_start)
    oA = _burst_onsets(sA, isi_gap)
    oB = _burst_onsets(sB, isi_gap)
    nA, nB = len(oA), len(oB)
    total_spikes = len(sA) + len(sB)

    out: Dict[str, object] = {
        "status": "silent",
        "freq": np.nan,
        "phase_onsets": np.array([], dtype=float),
        "onsets_A": oA,
        "onsets_B": oB,
        "n_burst_A": nA,
        "n_burst_B": nB,
        "n_spk_A": len(sA),
        "n_spk_B": len(sB),
        "cv_A": np.nan,
        "cv_B": np.nan,
        "cv_ref": np.nan,
        "alt_A_to_B": np.nan,
        "alt_B_to_A": np.nan,
        "alt_frac": np.nan,
        "balance": np.nan,
        "spk_per_burst_A": _mean_spikes_per_burst(sA, isi_gap),
        "spk_per_burst_B": _mean_spikes_per_burst(sB, isi_gap),
    }

    if total_spikes < 6:
        return out

    if max(nA, nB) == 0:
        out["status"] = "silent"
        return out

    balance = min(nA, nB) / max(nA, nB) if max(nA, nB) > 0 else 0.0
    out["balance"] = float(balance)

    # Both neurons must contribute to a genuine half-center rhythm.
    if nA < min_cycles or nB < min_cycles or balance < min_balance:
        out["status"] = "asymmetric" if total_spikes >= 20 else "silent"
        return out

    ibiA = np.diff(oA)
    ibiB = np.diff(oB)
    cvA = _cv(ibiA)
    cvB = _cv(ibiB)
    out["cv_A"] = cvA
    out["cv_B"] = cvB

    alt_A_to_B = _alternation_fraction(oA, oB)
    alt_B_to_A = _alternation_fraction(oB, oA)
    alt_frac = 0.5 * (alt_A_to_B + alt_B_to_A)
    out["alt_A_to_B"] = float(alt_A_to_B)
    out["alt_B_to_A"] = float(alt_B_to_A)
    out["alt_frac"] = float(alt_frac)

    if alt_frac < min_alt_frac:
        out["status"] = "tonic"
        return out

    # Prefer neuron A as reference if it is usable; otherwise use the cleaner side.
    if np.isfinite(cvA) and (not np.isfinite(cvB) or cvA <= cvB or len(oA) >= len(oB)):
        ref = oA
        ref_cv = cvA
    else:
        ref = oB
        ref_cv = cvB
    out["phase_onsets"] = ref
    out["cv_ref"] = ref_cv

    if not np.isfinite(ref_cv) or ref_cv > max_cv:
        out["status"] = "irregular"
        return out

    # Average frequency across valid sides. This is less noisy than one side only.
    freqs = []
    if len(ibiA) > 0 and np.mean(ibiA) > 0 and np.isfinite(cvA):
        freqs.append(1.0 / np.mean(ibiA))
    if len(ibiB) > 0 and np.mean(ibiB) > 0 and np.isfinite(cvB):
        freqs.append(1.0 / np.mean(ibiB))
    if not freqs:
        out["status"] = "irregular"
        return out

    out["freq"] = float(np.mean(freqs))
    out["status"] = "osc"
    return out


def phase_from_onsets(onsets: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Piecewise-linear unwrapped phase, advancing by 2π per reference cycle."""
    onsets = np.asarray(onsets, dtype=float)
    if len(onsets) < 2:
        return np.full_like(t_grid, np.nan, dtype=float)
    phases = 2 * np.pi * np.arange(len(onsets), dtype=float)
    return np.interp(t_grid, onsets, phases, left=np.nan, right=np.nan)


def phase_locking_metrics(
    onsets1: np.ndarray,
    onsets2: np.ndarray,
    dt: float = 0.01,
    n: int = 1,
    m: int = 1,
) -> Dict[str, float]:
    """Return n:m phase-locking value and residual phase-drift slope."""
    onsets1 = np.asarray(onsets1, dtype=float)
    onsets2 = np.asarray(onsets2, dtype=float)
    if len(onsets1) < 3 or len(onsets2) < 3:
        return {"R": np.nan, "dphi": np.nan, "slope": np.nan, "n_samples": 0}

    t0 = max(onsets1[0], onsets2[0])
    t1 = min(onsets1[-1], onsets2[-1])
    if t1 <= t0 + 2 * dt:
        return {"R": np.nan, "dphi": np.nan, "slope": np.nan, "n_samples": 0}

    t = np.arange(t0, t1, dt)
    phi1 = phase_from_onsets(onsets1, t)
    phi2 = phase_from_onsets(onsets2, t)
    mask = np.isfinite(phi1) & np.isfinite(phi2)
    if mask.sum() < 10:
        return {"R": np.nan, "dphi": np.nan, "slope": np.nan, "n_samples": int(mask.sum())}

    psi = n * phi1[mask] - m * phi2[mask]
    z = np.mean(np.exp(1j * psi))
    wrapped_mean = float(np.angle(z))
    R = float(np.abs(z))

    # Residual drift from unwrapped generalized phase difference.
    psi_unwrapped = np.unwrap(psi)
    if len(psi_unwrapped) >= 3:
        slope = float(np.polyfit(t[mask], psi_unwrapped, 1)[0])
    else:
        slope = np.nan

    return {"R": R, "dphi": wrapped_mean, "slope": slope, "n_samples": int(mask.sum())}


def best_nm_lock(
    onsets1: np.ndarray,
    onsets2: np.ndarray,
    max_order: int = 4,
    dt: float = 0.01,
) -> Dict[str, float]:
    """Find the strongest low-order n:m phase-locking relation."""
    best = {"n": 1, "m": 1, "R": np.nan, "dphi": np.nan, "slope": np.nan}
    for n in range(1, max_order + 1):
        for m in range(1, max_order + 1):
            metrics = phase_locking_metrics(onsets1, onsets2, dt=dt, n=n, m=m)
            R = metrics["R"]
            if np.isfinite(R) and (not np.isfinite(best["R"]) or R > best["R"]):
                best = {"n": n, "m": m, **metrics}
    return best


def evaluate_point(
    k2: float,
    g_inter: float,
    *,
    k1: float = 1.0,
    g_intra: float = 50e-9,
    T: float = 20.0,
    params: Optional[Dict[str, float]] = None,
    transient: Optional[float] = None,
    base_isi_gap: float = 0.15,
    freq_tol: float = FREQ_TOL,
    R_tol: float = R_TOL,
    slope_tol: float = SLOPE_TOL,
    nm_R_tol: float = 0.93,
    nm_max_order: int = 4,
) -> Dict[str, object]:
    """Run and classify one coupled-system point."""
    if transient is None:
        transient = max(2.0, 3.0 * max(k1, k2))
    transient = min(transient, 0.5 * T)

    spk = run_sim(T=T, k1=k1, k2=k2, g_intra=g_intra, g_inter=g_inter, params=params)
    a1 = analyze_segment(spk[0], spk[1], t_start=transient, isi_gap=base_isi_gap * k1)
    a2 = analyze_segment(spk[2], spk[3], t_start=transient, isi_gap=base_isi_gap * k2)

    out: Dict[str, object] = {
        "k2": float(k2),
        "g_inter": float(g_inter),
        "status1": a1["status"],
        "status2": a2["status"],
        "freq1": float(a1["freq"]),
        "freq2": float(a2["freq"]),
        "cv1": float(a1["cv_ref"]),
        "cv2": float(a2["cv_ref"]),
        "alt1": float(a1["alt_frac"]),
        "alt2": float(a2["alt_frac"]),
        "balance1": float(a1["balance"]),
        "balance2": float(a2["balance"]),
        "n_burst_1": int(a1["n_burst_A"] + a1["n_burst_B"]),
        "n_burst_2": int(a2["n_burst_A"] + a2["n_burst_B"]),
        "R_1_1": np.nan,
        "dphi_1_1": np.nan,
        "phase_slope_1_1": np.nan,
        "best_n": 1,
        "best_m": 1,
        "R_best": np.nan,
        "phase_slope_best": np.nan,
        "freq_ratio": np.nan,
        "locked_freq": np.nan,
        "regime": "no_osc",
    }

    if a1["status"] != "osc" or a2["status"] != "osc":
        out["lost_reason"] = f"{a1['status']}/{a2['status']}"
        return out

    f1 = float(a1["freq"])
    f2 = float(a2["freq"])
    f_mean = 0.5 * (f1 + f2)
    rel_diff = abs(f1 - f2) / f_mean if f_mean > 0 else np.inf
    out["freq_ratio"] = f1 / f2 if f2 > 0 else np.nan

    plv = phase_locking_metrics(a1["phase_onsets"], a2["phase_onsets"], dt=0.01, n=1, m=1)
    out["R_1_1"] = plv["R"]
    out["dphi_1_1"] = plv["dphi"]
    out["phase_slope_1_1"] = plv["slope"]

    best = best_nm_lock(a1["phase_onsets"], a2["phase_onsets"], max_order=nm_max_order, dt=0.01)
    out["best_n"] = int(best["n"])
    out["best_m"] = int(best["m"])
    out["R_best"] = best["R"]
    out["phase_slope_best"] = best["slope"]

    is_1_1 = (
        rel_diff < freq_tol
        and np.isfinite(plv["R"])
        and plv["R"] >= R_tol
        and np.isfinite(plv["slope"])
        and abs(plv["slope"]) <= slope_tol
    )
    if is_1_1:
        out["regime"] = "sync_1_1"
        out["locked_freq"] = f_mean
        return out

    # Optional low-order n:m lock. Require both high PLV and small generalized phase drift.
    if (
        np.isfinite(best["R"])
        and best["R"] >= nm_R_tol
        and np.isfinite(best["slope"])
        and abs(best["slope"]) <= slope_tol
        and not (best["n"] == 1 and best["m"] == 1)
    ):
        out["regime"] = "lock_n_m"
        return out

    out["regime"] = "drift"
    return out
