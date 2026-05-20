"""Create publication-oriented plots from sweep_data.npz."""

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
from text_equations import classification_text, model_equation_text, swept_parameters_text

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

REGIME_ORDER = ["sync_1_1", "drift", "no_osc"]
REGIME_LABEL = {
    "sync_1_1": "1:1 synchronized",
    "drift": "drift",
    "no_osc": "lost / invalid oscillation",
}
REGIME_COLORS = {
    "sync_1_1": "#2E7D32",
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
    ax.set_xlabel("target isolated mismatch Δf = f2_iso - f1_iso [Hz]\n(left: segment 2 slower; right: segment 2 faster)")
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
    ax.set_xlabel("measured isolated mismatch Δf = f2_iso - f1_iso [Hz]\n(left: slower segment 2; right: faster segment 2)")
    ax.set_ylabel("inter-segment inhibitory weight [nS, symlog]")
    ax.set_title(f"B. Empirical regime map (g_inter ≤ {y.max()/g_intra_nS:.2f} g_intra)")
    handles = [Patch(facecolor=REGIME_COLORS[r], label=REGIME_LABEL[r]) for r in REGIME_ORDER]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper right")

    # C. Locked frequency heatmap
    ax = fig.add_subplot(gs[1, 0])
    locked = data["locked_freq"].copy()
    locked[data["regime"] != "sync_1_1"] = np.nan
    pcm = ax.pcolormesh(Xe, Ye, locked, shading="auto")
    fig.colorbar(pcm, ax=ax, label="simulated resulting 1:1 frequency [Hz]")
    _add_zero_detuning_line(ax)
    _apply_symlog_coupling_axis(ax, y)
    ax.set_xlabel("measured isolated mismatch Δf = f2_iso - f1_iso [Hz]")
    ax.set_ylabel("inter-segment inhibitory weight [nS, symlog]")
    ax.set_title("C. Resulting frequency from simulated common-phase slope")

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
    ax.set_xlabel("measured isolated mismatch Δf = f2_iso - f1_iso [Hz]")
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

    labels = [r for r in ["sync_1_1", "drift", "no_osc"] if r in examples]
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

    The network schematic follows the clearer layout used in the minimal
    working example: two vertical half-centers, segment boxes, bold fixed
    intra-segment inhibition, and thinner crossed inter-segment inhibition.
    The right panel keeps the publication-oriented model equations and makes
    explicit which parameters are swept.
    """
    import matplotlib.patches as mpatches

    g_intra_nS = float(data.get("g_intra", np.array(50e-9))) * 1e9
    g_vals = np.asarray(data.get("g_inter_values", np.array([0.0])), dtype=float) * 1e9
    g_pos = g_vals[g_vals > 0]
    g_min = float(np.nanmin(g_pos)) if g_pos.size else 0.0
    g_max = float(np.nanmax(g_vals)) if g_vals.size else 0.0
    f1 = float(data.get("f_iso_1", np.array(np.nan)))
    df = np.asarray(data.get("target_delta_f", data.get("delta_f", np.array([]))), dtype=float)
    df_min = float(np.nanmin(df)) if df.size else np.nan
    df_max = float(np.nanmax(df)) if df.size else np.nan
    k1_abs = float(data.get("k1_abs", np.array(np.nan)))

    fig = plt.figure(figsize=(14.5, 8.8))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.18], wspace=0.18)

    # ------------------------------------------------------------------
    # Left panel: network diagram using the minimal-working-example layout
    # ------------------------------------------------------------------
    ax_sc = fig.add_subplot(gs[0, 0])
    ax_sc.set_xlim(0, 10)
    ax_sc.set_ylim(0, 12)
    ax_sc.set_aspect("equal")
    ax_sc.axis("off")
    ax_sc.set_title("A. Network topology", fontsize=12, fontweight="bold", pad=6)

    node_pos = {0: (2, 9), 1: (2, 6), 2: (8, 9), 3: (8, 6)}
    node_label = ["N0\nSeg1-A", "N1\nSeg1-B", "N2\nSeg2-A", "N3\nSeg2-B"]

    # Segment boxes, deliberately matching the supplied example style.
    for xs, lbl, subtitle in [
        ([0.5, 4.5], "Segment 1", "reference oscillator"),
        ([5.5, 9.5], "Segment 2", "detuned oscillator"),
    ]:
        rect = mpatches.FancyBboxPatch(
            (xs[0], 4.5), xs[1] - xs[0], 6.0,
            boxstyle="round,pad=0.3",
            linewidth=1.5,
            edgecolor="#999999",
            facecolor="none",
            zorder=1,
            linestyle="--",
        )
        ax_sc.add_patch(rect)
        ax_sc.text(np.mean(xs), 11.1, lbl, ha="center", va="bottom",
                   fontsize=9, color="#555555", fontstyle="italic", fontweight="bold")

    # Neurons.
    for n, (x, y) in node_pos.items():
        circ = plt.Circle((x, y), 0.9, color=NEURON_COLORS[n], zorder=4, ec="white", lw=1.5)
        ax_sc.add_patch(circ)
        ax_sc.text(x, y, node_label[n], ha="center", va="center",
                   fontsize=7.0, color="white", fontweight="bold", zorder=5)

    def draw_arrow(ax, p0, p1, color, lw, offset=(0, 0), style="solid", alpha=1.0):
        """Draw a connection in the same visual style as the supplied MWE."""
        x0 = p0[0] + offset[0]
        y0 = p0[1] + offset[1]
        x1 = p1[0] + offset[0]
        y1 = p1[1] + offset[1]
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=lw,
                shrinkA=10, shrinkB=10, mutation_scale=11,
                linestyle=style, alpha=alpha,
            ),
            zorder=2,
        )

    # Intra-segment arrows: fixed, strong, grey.
    intra_color = "#555555"
    draw_arrow(ax_sc, node_pos[0], node_pos[1], intra_color, 2.7, offset=(-0.32, 0))
    draw_arrow(ax_sc, node_pos[1], node_pos[0], intra_color, 2.7, offset=(0.32, 0))
    draw_arrow(ax_sc, node_pos[2], node_pos[3], intra_color, 2.7, offset=(0.32, 0))
    draw_arrow(ax_sc, node_pos[3], node_pos[2], intra_color, 2.7, offset=(-0.32, 0))

    # Inter-segment arrows: swept, crossed, orange.
    inter_color = "#FF8F00"
    draw_arrow(ax_sc, node_pos[0], node_pos[3], inter_color, 1.7, offset=(0, -0.22))
    draw_arrow(ax_sc, node_pos[3], node_pos[0], inter_color, 1.7, offset=(0, 0.22))
    draw_arrow(ax_sc, node_pos[1], node_pos[2], inter_color, 1.7, offset=(0, 0.22))
    draw_arrow(ax_sc, node_pos[2], node_pos[1], inter_color, 1.7, offset=(0, -0.22))

    # Legend for arrows. The detailed parameter values are kept in the legend and
    # right-hand methods panel to avoid cluttering the schematic itself.
    ax_sc.plot([], [], color=intra_color, lw=2.7, label=rf"Intra-segment inhibition ($g_{{intra}}={g_intra_nS:.0f}$ nS)")
    ax_sc.plot([], [], color=inter_color, lw=1.7, label=rf"Inter-segment inhibition ($g_{{inter}}={g_min:.4g}$–{g_max:.3g} nS, swept)")
    ax_sc.legend(fontsize=7.5, loc="lower center", framealpha=0.88, ncol=1,
                 bbox_to_anchor=(0.5, 0.14))

    ax_sc.text(
        5, 0.65,
        "All synapses are inhibitory; crossed inter-segment connections couple opposite phases",
        ha="center", va="bottom", fontsize=7.8, color="#333333",
        bbox=dict(boxstyle="round", fc="#EEEEEE", ec="#CCCCCC", lw=0.8),
    )

    # ------------------------------------------------------------------
    # Right panel: model equations and swept parameters
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    ax.text(0.0, 0.99, "B. Model equations and varied parameters", fontsize=12, weight="bold", va="top")

    eq_text = model_equation_text()
    ax.text(0.02, 0.88, eq_text, fontsize=10, va="top", family="serif", linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.75"))

    sweep_text = swept_parameters_text(
        f1=f1, k1_abs=k1_abs, df_min=df_min, df_max=df_max,
        g_min=g_min, g_max=g_max, g_intra_nS=g_intra_nS,
    )
    ax.text(0.02, 0.52, sweep_text, fontsize=10, va="top", linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.45", fc="#FFF7E6", ec=inter_color, lw=1.4))

    class_text = classification_text()
    ax.text(0.02, 0.03, class_text, fontsize=9.1, va="bottom", linespacing=1.3,
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
