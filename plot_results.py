"""Create publication-oriented plots from sweep_data.npz.

Version 3 plotting changes
--------------------------
- The y-axis is symlog-scaled to make very small inter-segment couplings visible
  while preserving the exact g=0 row.
- The x-axis is the measured isolated mismatch Δf, with a symmetric target Δf
  grid saved separately.
- The plots show k2 relative to the reference segment time scale, because this
  is the interpretable speed-up/slow-down control.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from analysis import analyze_segment, phase_from_onsets
from cpg_sim import run_sim

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

REGIME_ORDER = ["sync_1_1", "lock_n_m", "drift", "no_osc"]
REGIME_LABEL = {
    "sync_1_1": "1:1 synchronized",
    "lock_n_m": "n:m locked",
    "drift": "drift",
    "no_osc": "lost / invalid oscillation",
}
REGIME_COLORS = {
    "sync_1_1": "#2E7D32",
    "lock_n_m": "#8E24AA",
    "drift": "#1565C0",
    "no_osc": "#C62828",
}
NEURON_COLORS = ["#1565C0", "#E53935", "#2E7D32", "#8E24AA"]


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _sort_by_delta_f(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    order = np.argsort(data["delta_f"])
    out = dict(data)
    one_d = {
        "delta_f", "target_delta_f", "target_f2", "k2_values", "k2_rel_values",
        "f_iso_2", "status_iso_2", "cv_iso_2", "alt_iso_2",
    }
    for key, val in data.items():
        if key in one_d and hasattr(val, "ndim") and val.ndim == 1 and len(val) == len(order):
            out[key] = val[order]
        elif hasattr(val, "ndim") and val.ndim == 2 and val.shape[1] == len(order):
            out[key] = val[:, order]
    return out


def _edge_vector_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5])
    mid = 0.5 * (x[:-1] + x[1:])
    first = x[0] - (mid[0] - x[0])
    last = x[-1] + (x[-1] - mid[-1])
    return np.r_[first, mid, last]


def _edge_vector_symlog_positive(y: np.ndarray) -> np.ndarray:
    """Edges for non-negative values containing an exact zero row.

    y is expected to be [0, log-spaced positives...]. The returned edges are
    non-negative and suitable for pcolormesh followed by symlog y-scaling.
    """
    y = np.asarray(y, dtype=float)
    if len(y) == 1:
        return np.array([0.0, max(1e-12, y[0] + 1e-12)])
    if y[0] == 0 and np.all(y[1:] > 0):
        pos = y[1:]
        if len(pos) == 1:
            return np.array([0.0, pos[0] * 0.5, pos[0] * 1.5])
        mids = np.sqrt(pos[:-1] * pos[1:])
        first_pos_edge = pos[0] / np.sqrt(pos[1] / pos[0])
        last = pos[-1] * np.sqrt(pos[-1] / pos[-2])
        return np.r_[0.0, first_pos_edge, mids, last]
    return _edge_vector_linear(y)


def _regime_code(regime: np.ndarray) -> np.ndarray:
    code = np.full(regime.shape, len(REGIME_ORDER) - 1, dtype=int)
    for i, r in enumerate(REGIME_ORDER):
        code[regime == r] = i
    return code


def _apply_symlog_coupling_axis(ax, y_nS: np.ndarray) -> None:
    positive = y_nS[y_nS > 0]
    if len(positive):
        linthresh = float(max(positive.min(), 1e-6))
        ax.set_yscale("symlog", linthresh=linthresh, linscale=0.8)
        ax.set_ylim(0, float(y_nS.max() * 1.05))
    ax.grid(axis="y", which="both", lw=0.35, alpha=0.25)


def _add_zero_detuning_line(ax) -> None:
    ax.axvline(0.0, color="0.25", lw=0.8, ls=":")


def make_main_figure(data: Dict[str, np.ndarray], out_dir: Path) -> None:
    data = _sort_by_delta_f(data)
    x = np.asarray(data["delta_f"], dtype=float)
    y = np.asarray(data["g_inter_values"], dtype=float) * 1e9
    Xe = _edge_vector_linear(x)
    Ye = _edge_vector_symlog_positive(y)
    g_intra_nS = float(data["g_intra"]) * 1e9

    fig = plt.figure(figsize=(13, 10.5))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.25], hspace=0.36, wspace=0.30)

    # A. Isolated calibration
    ax = fig.add_subplot(gs[0, 0])
    f1 = float(data["f_iso_1"])
    k2_rel = data.get("k2_rel_values", data["k2_values"])
    target_f2 = data.get("target_f2", f1 / k2_rel)
    ax.plot(data["target_delta_f"], data["f_iso_2"], "o-", ms=3.5, label="measured isolated f2")
    ax.plot(data["target_delta_f"], target_f2, "--", lw=1.0, label="target f2 from symmetric Δf grid")
    ax.axhline(f1, color="0.3", lw=1, ls=":", label=f"measured f1={f1:.2f} Hz")
    bad = data["status_iso_2"] != "osc"
    if np.any(bad):
        ax.plot(data["target_delta_f"][bad], data["f_iso_2"][bad], "x", ms=7, label="invalid isolated oscillator")
    _add_zero_detuning_line(ax)
    ax.set_xlabel("target isolated mismatch Δf [Hz]")
    ax.set_ylabel("isolated frequency [Hz]")
    ax.set_title("A. Isolated-frequency calibration")
    ax.legend(frameon=False, fontsize=8)

    ax2 = ax.twiny()
    tick_locs = np.linspace(np.nanmin(data["target_delta_f"]), np.nanmax(data["target_delta_f"]), 5)
    tick_labels = []
    for loc in tick_locs:
        idx = int(np.nanargmin(np.abs(data["target_delta_f"] - loc)))
        tick_labels.append(f"{k2_rel[idx]:.2f}")
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(tick_locs)
    ax2.set_xticklabels(tick_labels)
    ax2.set_xlabel("segment-2 time scale relative to segment 1")

    # B. Regime map
    ax = fig.add_subplot(gs[0, 1])
    cmap = ListedColormap([REGIME_COLORS[r] for r in REGIME_ORDER])
    norm = BoundaryNorm(np.arange(len(REGIME_ORDER) + 1) - 0.5, cmap.N)
    ax.pcolormesh(Xe, Ye, _regime_code(data["regime"]), cmap=cmap, norm=norm, shading="auto")
    _add_zero_detuning_line(ax)
    _apply_symlog_coupling_axis(ax, y)
    ax.set_xlabel("measured isolated mismatch Δf = f1_iso - f2_iso [Hz]")
    ax.set_ylabel("inter-segment inhibitory weight [nS, symlog]")
    ax.set_title(f"B. Empirical regime map (g_inter ≤ {y.max()/g_intra_nS:.2f} g_intra)")
    handles = [Patch(facecolor=REGIME_COLORS[r], label=REGIME_LABEL[r]) for r in REGIME_ORDER]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper right")

    # C. Locked frequency heatmap
    ax = fig.add_subplot(gs[1, 0])
    locked = data["locked_freq"].copy()
    locked[data["regime"] != "sync_1_1"] = np.nan
    pcm = ax.pcolormesh(Xe, Ye, locked, shading="auto")
    fig.colorbar(pcm, ax=ax, label="locked 1:1 frequency [Hz]")
    _add_zero_detuning_line(ax)
    _apply_symlog_coupling_axis(ax, y)
    ax.set_xlabel("measured isolated mismatch Δf [Hz]")
    ax.set_ylabel("inter-segment inhibitory weight [nS, symlog]")
    ax.set_title("C. Resulting frequency inside 1:1 locking region")

    # D. PLV heatmap
    ax = fig.add_subplot(gs[1, 1])
    pcm = ax.pcolormesh(Xe, Ye, data["R_1_1"], shading="auto", vmin=0, vmax=1)
    fig.colorbar(pcm, ax=ax, label="1:1 phase-locking value R")
    try:
        ax.contour(x, y, data["regime"] == "sync_1_1", levels=[0.5], colors="k", linewidths=1.0)
    except Exception:
        pass
    _add_zero_detuning_line(ax)
    _apply_symlog_coupling_axis(ax, y)
    ax.set_xlabel("measured isolated mismatch Δf [Hz]")
    ax.set_ylabel("inter-segment inhibitory weight [nS, symlog]")
    ax.set_title("D. Phase-locking strength")

    fig.savefig(out_dir / "arnold_main.png", dpi=300)
    fig.savefig(out_dir / "arnold_main.pdf")
    plt.close(fig)


def _pick_examples(data: Dict[str, np.ndarray]) -> Dict[str, Tuple[int, int]]:
    examples: Dict[str, Tuple[int, int]] = {}
    regime = data["regime"]
    g = data["g_inter_values"]
    delta = data["delta_f"]

    idx = np.argwhere(regime == "sync_1_1")
    if len(idx):
        # Prefer small-to-moderate coupling away from trivial zero detuning.
        scores = []
        for j, i in idx:
            scores.append((abs(abs(delta[i]) - 0.5), abs(g[j] * 1e9 - np.nanmedian(g[g > 0] * 1e9)), j, i))
        _, _, j, i = min(scores)
        examples["sync_1_1"] = (j, i)

    idx = np.argwhere(regime == "drift")
    if len(idx):
        scores = []
        for j, i in idx:
            # Prefer a visible nonzero-coupling drift example with nontrivial detuning.
            scores.append((abs(g[j] * 1e9 - np.nanpercentile(g[g > 0] * 1e9, 25)), -abs(delta[i]), j, i))
        _, _, j, i = min(scores)
        examples["drift"] = (j, i)

    idx = np.argwhere(regime == "no_osc")
    if len(idx):
        scores = []
        for j, i in idx:
            scores.append((-g[j], abs(delta[i]), j, i))
        _, _, j, i = min(scores)
        examples["no_osc"] = (j, i)

    idx = np.argwhere(regime == "lock_n_m")
    if len(idx):
        scores = []
        for j, i in idx:
            scores.append((-data["R_best"][j, i] if np.isfinite(data["R_best"][j, i]) else 0, j, i))
        _, j, i = min(scores)
        examples["lock_n_m"] = (j, i)

    return examples


def _compute_phase_trace(spikes, k1_abs: float, k2_abs: float, T: float):
    transient = min(max(2.0, 3.0 * max(k1_abs, k2_abs)), 0.5 * T)
    a1 = analyze_segment(spikes[0], spikes[1], t_start=transient, isi_gap=0.15 * k1_abs)
    a2 = analyze_segment(spikes[2], spikes[3], t_start=transient, isi_gap=0.15 * k2_abs)
    t = np.arange(transient, T, 0.01)
    phi1 = phase_from_onsets(a1["phase_onsets"], t)
    phi2 = phase_from_onsets(a2["phase_onsets"], t)
    dphi = np.angle(np.exp(1j * (phi1 - phi2)))
    return t, dphi, a1, a2


def make_example_figure(data: Dict[str, np.ndarray], out_dir: Path, display_window_s: float = 2.0) -> None:
    examples = _pick_examples(data)
    if not examples:
        return

    labels = [r for r in ["sync_1_1", "drift", "no_osc", "lock_n_m"] if r in examples]
    nrows = len(labels)
    fig = plt.figure(figsize=(12, 3.2 * nrows))
    gs = gridspec.GridSpec(nrows, 3, figure=fig, width_ratios=[1.4, 1.0, 1.0], hspace=0.55, wspace=0.28)

    # Use the full sweep duration for re-running examples. Wide detuning sweeps
    # can include very slow oscillators; using a fixed 12 s run can falsely make
    # the slow segment look non-oscillatory. We still display only the last
    # display_window seconds so the traces remain readable.
    T = float(data["T"])
    display_window = min(float(display_window_s), T)
    params = {"Delta_w": float(data["base_Delta_w"]), "I_ext": float(data["base_I_ext"])}
    g_intra = float(data["g_intra"])
    k1_abs = float(data.get("k1_abs", 1.0))

    for row, regime in enumerate(labels):
        j, i = examples[regime]
        k2_abs = float(data["k2_values"][i])
        k2_rel = float(data.get("k2_rel_values", data["k2_values"])[i])
        g_inter = float(data["g_inter_values"][j])
        sim = run_sim(
            T=T,
            k1=k1_abs,
            k2=k2_abs,
            g_intra=g_intra,
            g_inter=g_inter,
            params=params,
            record=True,
            record_dt=1e-3,
        )
        spikes = sim["spikes"]

        title = (
            f"{REGIME_LABEL[regime]} | Δf={data['delta_f'][i]:.2f} Hz, "
            f"g={g_inter * 1e9:.3f} nS ({g_inter/g_intra:.3f} g_intra), "
            f"k2/k1={k2_rel:.2f}"
        )

        # Voltage traces: show a short readable window from the end of the run.
        ax = fig.add_subplot(gs[row, 0])
        t = sim["t"]
        u = sim["u"] * 1e3
        plot_t0 = max(0.0, T - display_window)
        win = t >= plot_t0
        for n in range(4):
            ax.plot(t[win], u[win, n] + 35 * n, lw=0.8, color=NEURON_COLORS[n], label=f"N{n}" if row == 0 else None)
        ax.set_title(title)
        ax.set_ylabel("Vm + offset [mV]")
        if row == nrows - 1:
            ax.set_xlabel("time [s]")
        if row == 0:
            ax.legend(frameon=False, ncol=4, fontsize=8)

        # Spike raster
        ax = fig.add_subplot(gs[row, 1])
        for n, s in enumerate(spikes):
            s_plot = s[(s >= plot_t0) & (s <= T)]
            ax.vlines(s_plot, n + 0.6, n + 1.4, color=NEURON_COLORS[n], lw=0.8)
        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels(["N0", "N1", "N2", "N3"])
        ax.set_ylim(0.5, 4.5)
        ax.set_title("spike raster")
        if row == nrows - 1:
            ax.set_xlabel("time [s]")

        # Phase difference
        ax = fig.add_subplot(gs[row, 2])
        try:
            tp, dphi, a1, a2 = _compute_phase_trace(spikes, k1_abs, k2_abs, T)
            phase_t0 = max(np.nanmin(tp) if len(tp) else 0.0, T - display_window)
            phase_win = tp >= phase_t0
            ax.plot(tp[phase_win], dphi[phase_win], lw=0.9, color="0.2")
            ax.set_ylim(-np.pi, np.pi)
            ax.set_yticks([-np.pi, 0, np.pi])
            ax.set_yticklabels(["-π", "0", "π"])
            ax.set_title(f"phase difference | f1={a1['freq']:.2f}, f2={a2['freq']:.2f} Hz")
        except Exception as exc:
            ax.text(0.5, 0.5, f"phase unavailable\n{exc}", ha="center", va="center")
        if row == nrows - 1:
            ax.set_xlabel("time [s]")

    fig.savefig(out_dir / "example_traces.png", dpi=300)
    fig.savefig(out_dir / "example_traces.pdf")
    plt.close(fig)


def make_architecture_figure(data: Dict[str, np.ndarray], out_dir: Path) -> None:
    """Draw the network architecture and model equations used in the sweep.

    The figure is intentionally self-contained: it shows the four-neuron
    architecture, distinguishes fixed intra-segment inhibition from swept
    inter-segment inhibition, and states the AdIF/synapse equations plus the
    time-dilation parameters used to impose isolated-frequency mismatch.
    """
    import matplotlib.patches as patches

    g_intra_nS = float(data.get("g_intra", np.array(50e-9))) * 1e9
    g_vals = np.asarray(data.get("g_inter_values", np.array([0.0])), dtype=float) * 1e9
    g_min = float(np.nanmin(g_vals[g_vals > 0])) if np.any(g_vals > 0) else 0.0
    g_max = float(np.nanmax(g_vals)) if len(g_vals) else 0.0
    f1 = float(data.get("f_iso_1", np.array(np.nan)))
    df = np.asarray(data.get("target_delta_f", data.get("delta_f", np.array([]))), dtype=float)
    df_min = float(np.nanmin(df)) if df.size else np.nan
    df_max = float(np.nanmax(df)) if df.size else np.nan
    k1_abs = float(data.get("k1_abs", np.array(np.nan)))

    fig = plt.figure(figsize=(13.5, 8.0))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.05, 1.25], wspace=0.18)

    # ------------------------------------------------------------------
    # Left panel: graph architecture
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    seg_boxes = [
        (0.6, 1.0, 3.6, 5.4, "Segment 1\nreference oscillator"),
        (5.8, 1.0, 3.6, 5.4, "Segment 2\ndetuned oscillator"),
    ]
    for x0, y0, w, h, lab in seg_boxes:
        rect = patches.FancyBboxPatch(
            (x0, y0), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
            ec="0.35", fc="0.98", lw=1.1
        )
        ax.add_patch(rect)
        ax.text(x0 + w/2, y0 + h - 0.35, lab, ha="center", va="top", fontsize=10, weight="bold")

    coords = {
        0: (2.4, 4.5), 1: (2.4, 2.3),
        2: (7.6, 4.5), 3: (7.6, 2.3),
    }
    labels = {0: "N0\nA", 1: "N1\nB", 2: "N2\nA", 3: "N3\nB"}
    for n, (x, y) in coords.items():
        circ = patches.Circle((x, y), 0.48, fc=NEURON_COLORS[n], ec="white", lw=1.5, alpha=0.95)
        ax.add_patch(circ)
        ax.text(x, y, labels[n], ha="center", va="center", fontsize=9, color="white", weight="bold")

    def inhib_arrow(a, b, color="0.2", lw=1.6, rad=0.0, label=None, label_xy=None):
        x1, y1 = coords[a]
        x2, y2 = coords[b]
        arrow = patches.FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-[", mutation_scale=12,
            connectionstyle=f"arc3,rad={rad}", lw=lw, color=color, shrinkA=35, shrinkB=35
        )
        ax.add_patch(arrow)
        if label:
            lx, ly = label_xy if label_xy else ((x1+x2)/2, (y1+y2)/2)
            ax.text(lx, ly, label, ha="center", va="center", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85), color=color)

    # fixed intra-segment mutual inhibition
    inhib_arrow(0, 1, color="0.25", lw=2.0, rad=0.18, label=r"$g_{\mathrm{intra}}$ fixed", label_xy=(1.35, 3.4))
    inhib_arrow(1, 0, color="0.25", lw=2.0, rad=0.18)
    inhib_arrow(2, 3, color="0.25", lw=2.0, rad=0.18, label=r"$g_{\mathrm{intra}}$ fixed", label_xy=(8.65, 3.4))
    inhib_arrow(3, 2, color="0.25", lw=2.0, rad=0.18)

    # swept crossed inter-segment inhibition
    sweep_color = "#C77700"
    inhib_arrow(0, 3, color=sweep_color, lw=2.3, rad=-0.08, label=r"$g_{\mathrm{inter}}$ swept", label_xy=(5.0, 2.55))
    inhib_arrow(3, 0, color=sweep_color, lw=2.3, rad=-0.08)
    inhib_arrow(1, 2, color=sweep_color, lw=2.3, rad=0.08, label=r"$g_{\mathrm{inter}}$ swept", label_xy=(5.0, 4.25))
    inhib_arrow(2, 1, color=sweep_color, lw=2.3, rad=0.08)

    ax.text(5.0, 6.72, "A. Network architecture", ha="center", va="top", fontsize=12, weight="bold")
    ax.text(5.0, 0.45,
            f"Sweep range: $g_{{inter}}$ = {g_min:.4g}–{g_max:.3g} nS; "
            f"$g_{{intra}}$ = {g_intra_nS:.1f} nS",
            ha="center", va="center", fontsize=9, color=sweep_color)

    # ------------------------------------------------------------------
    # Right panel: model and varied parameters
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    ax.text(0.0, 0.99, "B. Neuron, synapse and sweep parameters", fontsize=12, weight="bold", va="top")

    eq_text = (
        "Adaptive integrate-and-fire neuron\n"
        r"$\tau_{m,i}\frac{du_i}{dt}=-(u_i-E_{rest})-R_m w_i$" "\n"
        r"$\quad + R_m\left[I_{ext,i}+\sum_j g_{ij}(E_{inh}-u_i)\right]$" "\n"
        r"$\tau_{w,i}\frac{dw_i}{dt}=-w_i$" "\n"
        r"Spike if $u_i\geq E_{th}$: $u_i\leftarrow E_{rest}$, "
        r"$w_i\leftarrow w_i+\Delta w$, refractory for $t_{ref,i}$." "\n\n"
        "Conductance-based inhibition\n"
        r"$\tau_{syn,i}\frac{dg_{ij}}{dt}=-g_{ij}$" "\n"
        r"After delay $d_{ij}$, spike from $j$ increments "
        r"$g_{ij}\leftarrow g_{ij}+W_{ij}$."
    )
    ax.text(0.02, 0.88, eq_text, fontsize=10, va="top", family="serif", linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.75"))

    sweep_text = (
        "Parameters changed in the analysis\n"
        r"1. Frequency mismatch is imposed by segment-wise time dilation:" "\n"
        r"   $\tau_m,\tau_w,\tau_{syn},t_{ref},d \mapsto k_s(\cdot)$ "
        r"for segment $s$." "\n"
        fr"2. Segment 1 is calibrated to $f^{{iso}}_1\approx {f1:.2f}$ Hz "
        fr"($k_1$={k1_abs:.3g})." "\n"
        r"3. Segment 2 is assigned a symmetric target detuning " "\n"
        fr"   $\Delta f\in[{df_min:.2f},{df_max:.2f}]$ Hz, "
        r"where $\Delta f=f^{iso}_1-f^{iso}_2$." "\n"
        r"4. Crossed inter-segment inhibition $g_{inter}$ is swept " "\n"
        r"   logarithmically; $g_{intra}$ and neuron parameters are fixed."
    )
    ax.text(0.02, 0.52, sweep_text, fontsize=10, va="top", linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.45", fc="#FFF7E6", ec="#C77700", lw=1.4))

    classification_text = (
        "Point classification\n"
        r"A grid point is called 1:1 synchronized only if both " "\n"
        r"segments remain oscillatory, their measured frequencies " "\n"
        r"match within tolerance, the 1:1 phase-locking value is high, " "\n"
        r"and the unwrapped phase difference has negligible residual drift. " "\n"
        r"Otherwise the point is classified as drift, n:m locking, " "\n"
        r"or lost/invalid oscillation."
    )
    ax.text(0.02, 0.02, classification_text, fontsize=9, va="bottom", linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.4", fc="0.97", ec="0.82"))

    fig.savefig(out_dir / "network_architecture_equations.png", dpi=300)
    fig.savefig(out_dir / "network_architecture_equations.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="results/sweep_data.npz")
    parser.add_argument("--out", default=None)
    parser.add_argument("--example-window-s", type=float, default=2.0, help="Displayed time window for example traces")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out) if args.out else data_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_npz(data_path)
    make_main_figure(data, out_dir)
    make_architecture_figure(data, out_dir)
    make_example_figure(data, out_dir, display_window_s=args.example_window_s)
    print(f"Saved figures in {out_dir}")


if __name__ == "__main__":
    main()
