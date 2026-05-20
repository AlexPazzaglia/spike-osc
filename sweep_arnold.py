"""Run the isolated calibration and coupled 2D Arnold-tongue sweep."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from analysis import analyze_segment, evaluate_point
from cpg_sim import run_sim


def _measure_segment_frequency(
    *,
    k_abs: float,
    T: float,
    params: Dict[str, float],
    g_intra: float,
    base_isi_gap: float = 0.15,
) -> Dict[str, object]:
    """Measure one isolated segment run with both segments at the same scale."""
    spk = run_sim(T=T, k1=k_abs, k2=k_abs, g_intra=g_intra, g_inter=0.0, params=params)
    transient = min(max(2.0, 3.0 * k_abs), 0.5 * T)
    return analyze_segment(spk[0], spk[1], t_start=transient, isi_gap=base_isi_gap * k_abs)


def estimate_reference_time_scale(
    *,
    target_f1_hz: float,
    T: float,
    params: Dict[str, float],
    g_intra: float,
) -> Tuple[float, float, float]:
    """Return k1_abs that makes segment 1 approximately target_f1_hz.

    Because frequency control is implemented by time dilation, measuring the
    isolated frequency at k=1 is enough to estimate the scale needed for the
    desired reference frequency.
    """
    a = _measure_segment_frequency(k_abs=1.0, T=T, params=params, g_intra=g_intra)
    if a["status"] != "osc" or not np.isfinite(a["freq"]):
        raise RuntimeError(
            "The baseline oscillator at k=1 is not a valid isolated oscillator. "
            f"Status={a['status']}, freq={a['freq']}"
        )
    f_at_k1 = float(a["freq"])
    k1_abs = f_at_k1 / target_f1_hz

    # Verify the estimate rather than trusting the exact time-dilation ideal.
    verify = _measure_segment_frequency(k_abs=k1_abs, T=T, params=params, g_intra=g_intra)
    if verify["status"] != "osc" or not np.isfinite(verify["freq"]):
        raise RuntimeError(
            "The estimated reference time scale does not produce a valid isolated oscillator. "
            f"k1_abs={k1_abs:.4g}, status={verify['status']}, freq={verify['freq']}"
        )
    return float(k1_abs), f_at_k1, float(verify["freq"])


def default_grids(
    *,
    target_f1_hz: float,
    k1_abs: float,
    f1_actual_hz: float,
    g_intra: float,
    delta_f_max_hz: float | None = None,
    n_delta: int | None = None,
    n_g: int | None = None,
    g_min_nS: float = 0.002,
    g_max_frac: float = 0.5,
    min_f2_hz: float = 0.25,
    clip_invalid_delta: bool = True,
) -> Dict[str, np.ndarray | float]:
    """Build a symmetric Δf grid and a log-spaced low-coupling grid.

    Positive Δf means segment 2 is intrinsically faster than the reference:
    f2 = f1 + Δf. Therefore the slow-side bound occurs at negative Δf. In
    practice, we enforce a configurable minimum f2 to avoid edge points where
    the simulation contains too few cycles to classify reliably.
    """
    if delta_f_max_hz is None:
        # Keep f2 positive and avoid the extremely slow edge by default.
        delta_f_max_hz = 2.4
    if n_delta is None:
        n_delta = 61
    if n_g is None:
        n_g = 45

    # With the convention Δf = f2 - f1, the negative Δf side implies
    # f2 = f1 + Δf. Enforce an absolute minimum f2 rather than an
    # arbitrary fractional limit.
    if min_f2_hz <= 0:
        raise ValueError("min_f2_hz must be positive")
    max_allowed = f1_actual_hz - min_f2_hz
    if max_allowed <= 0:
        raise ValueError(
            f"min_f2_hz={min_f2_hz:.3f} is not smaller than f1={f1_actual_hz:.3f} Hz."
        )
    if delta_f_max_hz > max_allowed:
        msg = (
            f"Requested delta_f_max_hz={delta_f_max_hz:.4f} Hz would make the "
            f"slow oscillator target f2={f1_actual_hz - delta_f_max_hz:.4f} Hz, "
            f"below min_f2_hz={min_f2_hz:.4f} Hz. Maximum allowed is {max_allowed:.4f} Hz."
        )
        if clip_invalid_delta:
            print("WARNING: " + msg)
            print(f"Clipping delta_f_max_hz to {max_allowed:.4f} Hz.")
            delta_f_max_hz = max_allowed
        else:
            raise ValueError(msg)

    target_delta_f = np.linspace(-delta_f_max_hz, delta_f_max_hz, int(n_delta))
    target_f2 = f1_actual_hz + target_delta_f
    if np.any(target_f2 <= 0):
        raise ValueError("Target f2 contains non-positive frequencies; reduce delta_f_max_hz.")

    # Time-dilation approximation: f(k) ≈ f(k1_abs) * k1_abs / k2_abs.
    k2_abs_values = k1_abs * f1_actual_hz / target_f2
    k2_rel_values = k2_abs_values / k1_abs

    g_max = g_max_frac * g_intra
    if g_max <= 0:
        raise ValueError("g_max must be positive")
    if g_min_nS <= 0:
        raise ValueError("g_min_nS must be positive")
    g_min = g_min_nS * 1e-9
    if g_min >= g_max:
        raise ValueError("g_min_nS must be smaller than g_max")

    # Include exact zero, then logarithmic spacing. This gives much better
    # resolution near the tongue tip and avoids spending samples close to g_intra.
    g_values = np.concatenate([[0.0], np.geomspace(g_min, g_max, int(n_g) - 1)])

    return {
        "target_delta_f": target_delta_f,
        "target_f2": target_f2,
        "k2_values": k2_abs_values,
        "k2_rel_values": k2_rel_values,
        "g_inter_values": g_values,
        "used_delta_f_max_hz": np.array(delta_f_max_hz),
        "min_f2_hz": np.array(min_f2_hz),
    }


def recommended_duration(
    *,
    k1_abs: float,
    k2_values: np.ndarray,
    target_f2: np.ndarray,
    min_eval_cycles: int = 8,
    buffer_s: float = 2.0,
) -> float:
    """Return a conservative duration for the slowest target oscillator.

    The analysis discards an initial transient of about 3*max(k1,k2), so wide
    detuning sweeps need longer simulations at the slow edge. This helper keeps
    the post-transient interval long enough to contain several cycles of the
    slowest oscillator.
    """
    k_max = float(np.nanmax(np.r_[k1_abs, np.asarray(k2_values, dtype=float)]))
    f_min = float(np.nanmin(np.asarray(target_f2, dtype=float)))
    transient_est = max(2.0, 3.0 * k_max)
    return float(transient_est + min_eval_cycles / f_min + buffer_s)


def calibrate_isolated(
    k2_values: np.ndarray,
    *,
    k1_abs: float,
    T: float,
    params: Dict[str, float],
    g_intra: float,
    base_isi_gap: float = 0.15,
) -> Dict[str, np.ndarray]:
    """Measure isolated frequencies and oscillator validity for each k2."""
    f_iso_2 = np.full(len(k2_values), np.nan)
    status_iso_2 = np.empty(len(k2_values), dtype=object)
    cv_iso_2 = np.full(len(k2_values), np.nan)
    alt_iso_2 = np.full(len(k2_values), np.nan)

    spk0 = run_sim(T=T, k1=k1_abs, k2=k1_abs, g_intra=g_intra, g_inter=0.0, params=params)
    trans0 = min(max(2.0, 3.0 * k1_abs), 0.5 * T)
    a1 = analyze_segment(spk0[0], spk0[1], t_start=trans0, isi_gap=base_isi_gap * k1_abs)
    f_iso_1 = float(a1["freq"])
    status_iso_1 = str(a1["status"])
    cv_iso_1 = float(a1["cv_ref"])
    alt_iso_1 = float(a1["alt_frac"])

    for i, k2_abs in enumerate(k2_values):
        spk = run_sim(T=T, k1=k1_abs, k2=float(k2_abs), g_intra=g_intra, g_inter=0.0, params=params)
        trans = min(max(2.0, 3.0 * max(k1_abs, float(k2_abs))), 0.5 * T)
        a2 = analyze_segment(spk[2], spk[3], t_start=trans, isi_gap=base_isi_gap * float(k2_abs))
        f_iso_2[i] = float(a2["freq"])
        status_iso_2[i] = str(a2["status"])
        cv_iso_2[i] = float(a2["cv_ref"])
        alt_iso_2[i] = float(a2["alt_frac"])

    return {
        "f_iso_1": np.array(f_iso_1),
        "status_iso_1": np.array(status_iso_1),
        "cv_iso_1": np.array(cv_iso_1),
        "alt_iso_1": np.array(alt_iso_1),
        "f_iso_2": f_iso_2,
        "status_iso_2": np.array(status_iso_2, dtype=str),
        "cv_iso_2": cv_iso_2,
        "alt_iso_2": alt_iso_2,
    }


def sweep(
    k2_values: np.ndarray,
    g_values: np.ndarray,
    *,
    k1_abs: float,
    T: float,
    params: Dict[str, float],
    g_intra: float,
) -> Dict[str, np.ndarray]:
    Ng, Nk = len(g_values), len(k2_values)

    str_fields = ["regime", "status1", "status2", "lost_reason"]
    num_fields = [
        "freq1", "freq2", "cv1", "cv2", "alt1", "alt2", "balance1", "balance2",
        "n_burst_1", "n_burst_2", "R_1_1", "dphi_1_1", "phase_slope_1_1",
        "freq_ratio", "locked_freq",
    ]
    out: Dict[str, np.ndarray] = {}
    for field in str_fields:
        out[field] = np.full((Ng, Nk), "", dtype="U32")
    for field in num_fields:
        out[field] = np.full((Ng, Nk), np.nan, dtype=float)

    t0 = time.time()
    # Repeated Numba calls allocate many short-lived arrays. Python's cyclic
    # garbage collector can occasionally pause for a very long time during wide
    # sweeps, even though these objects are refcounted and not cyclic. Disable
    # cyclic GC during the inner sweep and re-enable it before returning.
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()

    for j, g in enumerate(g_values):
        for i, k2_abs in enumerate(k2_values):
            r = evaluate_point(
                k1=k1_abs,
                k2=float(k2_abs),
                g_inter=float(g),
                T=T,
                g_intra=g_intra,
                params=params,
            )
            for field in str_fields:
                out[field][j, i] = str(r.get(field, ""))
            for field in num_fields:
                val = r.get(field, np.nan)
                try:
                    out[field][j, i] = float(val)
                except Exception:
                    out[field][j, i] = np.nan

        done = (j + 1) * Nk
        total = Ng * Nk
        elapsed = time.time() - t0
        eta = elapsed * (total / max(done, 1) - 1)
        row_regimes, row_counts = np.unique(out["regime"][j], return_counts=True)
        row_summary = ", ".join(f"{r}:{c}" for r, c in zip(row_regimes, row_counts))
        print(
            f"g row {j + 1:02d}/{Ng} | g={g * 1e9:8.4f} nS "
            f"({g / g_intra:5.3f} g_intra) | {done:5d}/{total} sims | "
            f"elapsed={elapsed:6.1f}s | eta={eta:6.1f}s | {row_summary}",
            flush=True,
        )
    if gc_was_enabled:
        gc.enable()
    return out


def save_point_table(path: Path, data: Dict[str, np.ndarray]) -> None:
    """Save a long-form CSV table, one row per sweep point."""
    k2_values = data["k2_values"]
    k2_rel_values = data["k2_rel_values"]
    g_values = data["g_inter_values"]
    delta_f = data["delta_f"]
    target_delta_f = data["target_delta_f"]
    fields = [
        "regime", "status1", "status2", "freq1", "freq2", "locked_freq",
        "R_1_1", "phase_slope_1_1", "cv1", "cv2", "alt1", "alt2", "balance1", "balance2",
    ]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "k1_abs", "k2_abs", "k2_relative_to_k1", "target_delta_f_Hz", "measured_delta_f_Hz",
            "g_inter_S", "g_inter_nS", "g_inter_frac_of_g_intra", *fields,
        ])
        for j, g in enumerate(g_values):
            for i, k2_abs in enumerate(k2_values):
                writer.writerow([
                    float(data["k1_abs"]), float(k2_abs), float(k2_rel_values[i]),
                    float(target_delta_f[i]), float(delta_f[i]), float(g), float(g * 1e9),
                    float(g / data["g_intra"]), *[data[field][j, i] for field in fields],
                ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results", help="Output directory")
    parser.add_argument("--T", type=float, default=None, help="Simulation duration in seconds")
    parser.add_argument("--target-f1-Hz", type=float, default=3.0, help="Target isolated frequency of segment 1")
    parser.add_argument("--delta-f-max-Hz", type=float, default=None, help="Symmetric detuning range for Δf=f2_iso-f1_iso: [-D, +D] Hz")
    parser.add_argument("--n-delta", type=int, default=None, help="Number of detuning samples")
    parser.add_argument("--n-g", type=int, default=None, help="Number of coupling samples including zero")
    parser.add_argument("--g-min-nS", type=float, default=0.002, help="Smallest nonzero inter-segment weight")
    parser.add_argument("--g-max-frac", type=float, default=0.5, help="Maximum g_inter as fraction of g_intra")
    parser.add_argument("--min-f2-Hz", type=float, default=0.25, help="Minimum target isolated frequency allowed for segment 2 on the slow side")
    parser.add_argument("--strict-delta-limit", action="store_true", help="Raise an error instead of clipping if the requested detuning makes f2 too slow")
    parser.add_argument("--auto-T", dest="auto_T", action="store_true", default=True, help="Automatically increase T for wide detuning sweeps when --T is not explicitly set")
    parser.add_argument("--no-auto-T", dest="auto_T", action="store_false", help="Disable automatic duration adjustment")
    parser.add_argument("--min-eval-cycles", type=int, default=8, help="Minimum post-transient cycles targeted by automatic T selection")
    parser.add_argument("--base-k1", type=float, default=None, help="Override automatic reference time scale")
    parser.add_argument("--g-intra-nS", type=float, default=50.0, help="Intra-segment weight in nS")
    parser.add_argument("--Delta-w-pA", type=float, default=25.0, help="Adaptation increment in pA")
    parser.add_argument("--I-ext-pA", type=float, default=240.0, help="External current in pA")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    T = float(args.T) if args.T is not None else 20.0
    g_intra = args.g_intra_nS * 1e-9
    params = {"Delta_w": args.Delta_w_pA * 1e-12, "I_ext": args.I_ext_pA * 1e-12}

    print("=== Reference oscillator calibration ===")
    if args.base_k1 is None:
        k1_abs, f_at_k1_raw, f1_est = estimate_reference_time_scale(
            target_f1_hz=args.target_f1_Hz,
            T=T,
            params=params,
            g_intra=g_intra,
        )
        print(f"f(k=1)={f_at_k1_raw:.4f} Hz -> estimated k1_abs={k1_abs:.5f}")
    else:
        k1_abs = float(args.base_k1)
        a_ref = _measure_segment_frequency(k_abs=k1_abs, T=T, params=params, g_intra=g_intra)
        f1_est = float(a_ref["freq"])
        print(f"using user-specified k1_abs={k1_abs:.5f}")

    print(f"verified reference f1_iso~{f1_est:.4f} Hz")

    grids = default_grids(
        target_f1_hz=args.target_f1_Hz,
        k1_abs=k1_abs,
        f1_actual_hz=f1_est,
        g_intra=g_intra,
        delta_f_max_hz=args.delta_f_max_Hz,
        n_delta=args.n_delta,
        n_g=args.n_g,
        g_min_nS=args.g_min_nS,
        g_max_frac=args.g_max_frac,
        min_f2_hz=args.min_f2_Hz,
        clip_invalid_delta=not args.strict_delta_limit,
    )
    k2_values = np.asarray(grids["k2_values"], dtype=float)
    k2_rel_values = np.asarray(grids["k2_rel_values"], dtype=float)
    g_values = np.asarray(grids["g_inter_values"], dtype=float)
    target_delta_f = np.asarray(grids["target_delta_f"], dtype=float)
    target_f2 = np.asarray(grids["target_f2"], dtype=float)

    rec_T = recommended_duration(
        k1_abs=k1_abs,
        k2_values=k2_values,
        target_f2=target_f2,
        min_eval_cycles=args.min_eval_cycles,
    )
    if args.T is None and args.auto_T and T < rec_T:
        print(
            f"Auto-increasing T from {T:.2f}s to {rec_T:.2f}s so the slowest "
            f"target oscillator f2~{np.nanmin(target_f2):.3f} Hz has enough cycles."
        )
        T = rec_T
    elif T < rec_T:
        print(
            f"WARNING: T={T:.2f}s may be too short for the slowest target "
            f"oscillator f2~{np.nanmin(target_f2):.3f} Hz. Recommended T~{rec_T:.2f}s "
            f"or use --auto-T without an explicit --T."
        )

    print("\n=== Isolated calibration across symmetric Δf grid ===")
    calib = calibrate_isolated(k2_values, k1_abs=k1_abs, T=T, params=params, g_intra=g_intra)
    f_iso_1 = float(calib["f_iso_1"])
    f_iso_2 = calib["f_iso_2"]
    delta_f = f_iso_2 - f_iso_1
    print(f"f_iso_1 = {f_iso_1:.4f} Hz | status={calib['status_iso_1']}")
    print(f"f_iso_2 valid: {np.sum(calib['status_iso_2'] == 'osc')}/{len(k2_values)}")
    print(f"target Δf range: {np.nanmin(target_delta_f):.4f} .. {np.nanmax(target_delta_f):.4f} Hz")
    print(f"measured Δf range: {np.nanmin(delta_f):.4f} .. {np.nanmax(delta_f):.4f} Hz")
    print(f"k2_abs range: {np.nanmin(k2_values):.4f} .. {np.nanmax(k2_values):.4f}")
    print(f"g_inter range: {np.nanmin(g_values)*1e9:.4f} .. {np.nanmax(g_values)*1e9:.4f} nS")
    print(f"max g_inter/g_intra = {np.nanmax(g_values)/g_intra:.3f}")

    print("\n=== Coupled sweep ===")
    sw = sweep(k2_values, g_values, k1_abs=k1_abs, T=T, params=params, g_intra=g_intra)

    data = {
        "k1_abs": np.array(k1_abs),
        "k2_values": k2_values,
        "k2_rel_values": k2_rel_values,
        "target_delta_f": target_delta_f,
        "target_f2": target_f2,
        "used_delta_f_max_hz": np.asarray(grids.get("used_delta_f_max_hz", np.nan)),
        "min_f2_hz": np.asarray(grids.get("min_f2_hz", args.min_f2_Hz)),
        "g_inter_values": g_values,
        "delta_f": delta_f,
        "T": np.array(T),
        "target_f1_Hz": np.array(args.target_f1_Hz),
        "g_intra": np.array(g_intra),
        "base_Delta_w": np.array(params["Delta_w"]),
        "base_I_ext": np.array(params["I_ext"]),
        **calib,
        **sw,
    }

    npz_path = out_dir / "sweep_data.npz"
    np.savez(npz_path, **data)
    save_point_table(out_dir / "sweep_points.csv", data)

    regimes, counts = np.unique(sw["regime"], return_counts=True)
    summary = {
        "T": T,
        "recommended_T_for_grid": rec_T,
        "auto_T_used": bool(args.T is None and args.auto_T),
        "target_f1_Hz": args.target_f1_Hz,
        "verified_f1_iso_Hz": f_iso_1,
        "k1_abs": k1_abs,
        "n_delta": int(len(k2_values)),
        "n_g": int(len(g_values)),
        "n_simulations": int(len(k2_values) * len(g_values)),
        "target_delta_f_min_Hz": float(np.nanmin(target_delta_f)),
        "target_delta_f_max_Hz": float(np.nanmax(target_delta_f)),
        "measured_delta_f_min_Hz": float(np.nanmin(delta_f)),
        "measured_delta_f_max_Hz": float(np.nanmax(delta_f)),
        "g_min_nonzero_nS": float(g_values[1] * 1e9) if len(g_values) > 1 else None,
        "g_max_nS": float(np.nanmax(g_values) * 1e9),
        "g_max_frac_of_g_intra": float(np.nanmax(g_values) / g_intra),
        "min_f2_Hz": float(args.min_f2_Hz),
        "used_delta_f_max_Hz": float(np.nanmax(np.abs(target_delta_f))),
        "regime_counts": {str(k): int(v) for k, v in zip(regimes, counts)},
        "params": params,
        "g_intra_S": g_intra,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {npz_path}")
    print(f"Saved: {out_dir / 'sweep_points.csv'}")
    print(f"Saved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
    # Wide sweeps can leave large Numba/Python object graphs that occasionally
    # make interpreter shutdown hang on some platforms. All output files are
    # closed by this point, so use a hard process exit after flushing streams.
    import os as _os
    import sys as _sys
    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(0)
