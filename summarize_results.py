"""Summarize and interpret one completed Arnold/regime sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REGIME_ORDER = ["sync_1_1", "lock_n_m", "drift", "no_osc"]


def load(path: Path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def first_sync_boundary(data):
    """Return minimal nonzero coupling required for 1:1 locking per detuning."""
    delta = np.asarray(data["delta_f"], dtype=float)
    g = np.asarray(data["g_inter_values"], dtype=float)
    regime = data["regime"]
    boundary = []
    for i, df in enumerate(delta):
        idx = np.where(regime[:, i] == "sync_1_1")[0]
        if len(idx) == 0:
            boundary.append((df, np.nan, np.nan))
        else:
            j = idx[0]
            boundary.append((df, g[j], g[j] / float(data["g_intra"])))
    return np.array(boundary, dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="results/sweep_data.npz")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    path = Path(args.data)
    out_dir = Path(args.out) if args.out else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load(path)

    regime = data["regime"]
    total = regime.size
    counts = {r: int(np.sum(regime == r)) for r in REGIME_ORDER if np.any(regime == r)}
    perc = {r: 100 * c / total for r, c in counts.items()}

    boundary = first_sync_boundary(data)
    np.savetxt(
        out_dir / "sync_boundary.csv",
        boundary,
        delimiter=",",
        header="delta_f_Hz,min_g_inter_S,min_g_inter_frac_of_g_intra",
        comments="",
    )

    # Width of 1:1 locking at each coupling row.
    delta = np.asarray(data["delta_f"], dtype=float)
    g = np.asarray(data["g_inter_values"], dtype=float)
    widths = []
    for j, gj in enumerate(g):
        mask = regime[j] == "sync_1_1"
        if np.any(mask):
            widths.append((gj, gj / float(data["g_intra"]), float(delta[mask].min()), float(delta[mask].max()), float(delta[mask].max() - delta[mask].min())))
        else:
            widths.append((gj, gj / float(data["g_intra"]), np.nan, np.nan, 0.0))
    widths = np.array(widths, dtype=float)
    np.savetxt(
        out_dir / "sync_width_by_coupling.csv",
        widths,
        delimiter=",",
        header="g_inter_S,g_inter_frac_of_g_intra,delta_f_min_locked_Hz,delta_f_max_locked_Hz,locked_width_Hz",
        comments="",
    )

    summary = {
        "n_points": int(total),
        "regime_counts": counts,
        "regime_percentages": {r: round(v, 2) for r, v in perc.items()},
        "f_iso_1_Hz": float(data["f_iso_1"]),
        "target_f1_Hz": float(data.get("target_f1_Hz", np.nan)),
        "delta_f_min_Hz": float(np.nanmin(delta)),
        "delta_f_max_Hz": float(np.nanmax(delta)),
        "g_max_nS": float(np.nanmax(g) * 1e9),
        "g_max_frac_of_g_intra": float(np.nanmax(g) / float(data["g_intra"])),
        "has_no_oscillation_region": bool(np.any(regime == "no_osc")),
        "has_nm_locking_region": bool(np.any(regime == "lock_n_m")),
        "notes": [
            "Use sync_boundary.csv to approximate the empirical Arnold-tongue boundary.",
            "Use sync_width_by_coupling.csv to quantify how locking width expands with coupling.",
            "If sync_1_1 remains dominant after widening Δf, the oscillator pair is strongly entrainable under crossed inhibition rather than merely weakly phase-coupled.",
        ],
    }
    with (out_dir / "interpretation_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved {out_dir / 'sync_boundary.csv'}")
    print(f"Saved {out_dir / 'sync_width_by_coupling.csv'}")
    print(f"Saved {out_dir / 'interpretation_summary.json'}")


if __name__ == "__main__":
    main()
