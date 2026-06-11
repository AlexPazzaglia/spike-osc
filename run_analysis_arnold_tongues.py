"""
Arnold tongues for the 2-segment spiking CPG.

Sweep (detuning, inter-segment inhibition), measure phase locking between
the two segments via the Phase Locking Value (PLV) of phi_2 - phi_1, and
plot the resulting tongue map.

The detuning axis is the absolute difference of the uncoupled burst
frequencies f1 - f2 (Hz). For each t_scale, an uncoupled run (J_inter = 0)
is performed before the coupled one to measure (f1, f2). The grid is
chosen symmetrically in relative detuning, with an odd number of points,
so that f1 - f2 = 0 falls at the centre of its bin.

Simulations are dispatched in parallel across processes.

Generalised PLV for p:q locking. At seg-1 burst onsets, phi_1 = 0 mod 2pi,
so |<exp(i (q phi_1 - p phi_2))>| reduces to |<exp(i p phi_2)>| — sensitive
to p but blind to q. The symmetric measurement at seg-2 onsets is sensitive
to q. Taking the min of the two scores high only for true p:q locking.

PLV_{p,q} = min( |<exp(i p phi_2)>|_{seg-1 onsets},
                 |<exp(i q phi_1)>|_{seg-2 onsets} )   in [0, 1]
    1 = perfect p:q phase locking, 0 = no locking.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import brian2 as b2
import matplotlib.pyplot as plt

from cpg_simulation import SpikingCPG


# Candidate p:q locking ratios (low p, q).
PQ_LIST = [(1,1), (1,2), (2,1), (1,3), (3,1), (2,3), (3,2)]


# -------------------------------------------------------------------------
# Single runs
# -------------------------------------------------------------------------

def _build_cpg(t_scale: float, J_inter: float):
    """Build a CPG with the given inter-segment weight."""
    cpg = SpikingCPG(
        t_scale1 = 1.0,
        t_scale2 = t_scale,
        record   = False,
        callback = False,
    )
    cpg._assign_synaptic_weights(J_inter = J_inter * b2.pA)
    return cpg


def _burst_plv(
    bst1  : np.ndarray,
    bst2  : np.ndarray,
    t_skip: float,
    p     : int = 1,
    q     : int = 1,
):
    """Generalised PLV for p:q locking.

    The inner helper samples the phase of b2 at b1 burst onsets and returns
    |<exp(i n phi)>|. Calling it both ways (n=p one way, n=q the other) and
    taking the min disambiguates p from q.
    """

    def _side(b1, b2, n):
        b1 = b1[b1 >= t_skip]
        if len(b1) < 2 or len(b2) < 2:
            return np.nan

        # Phase of b2 at each b1 onset: linear between bracketing bursts
        idx     = np.searchsorted(b2, b1) - 1
        ok      = (idx >= 0) & (idx < len(b2) - 1)
        idx, t  = idx[ok], b1[ok]
        if len(t) < 2:
            return np.nan

        t0, t1 = b2[idx], b2[idx + 1]
        phi    = 2 * np.pi * (t - t0) / (t1 - t0)
        return float(np.abs(np.mean(np.exp(1j * n * phi))))

    return float(np.minimum(_side(bst1, bst2, p), _side(bst2, bst1, q)))


def measure_freqs(
    t_scale : float,
    duration: float = 50.0,
    t_skip  : float = 10.0,
):
    """Uncoupled run: return (f1, f2) from mean burst periods."""

    cpg = _build_cpg(t_scale, J_inter = 0.0)
    cpg.run(duration * b2.second)

    bst_on = cpg.sim_out['bursts_onsets']
    freqs  = []
    for st in (bst_on[0], bst_on[2]):
        st = st[st >= t_skip]
        f  = 1.0 / np.mean(np.diff(st)) if len(st) >= 2 else np.nan
        freqs.append(f)
    return freqs[0], freqs[1]


def run_one(
    t_scale : float,
    J_inter : float,
    duration: float = 50.0,
    t_skip  : float = 10.0,
):
    """Coupled run: return PLV between segment 1 and segment 2 for each (p, q)."""

    cpg = _build_cpg(t_scale, J_inter)
    cpg.run(duration * b2.second)

    bst_on = cpg.sim_out['bursts_onsets']
    return np.array([
        _burst_plv(bst_on[0], bst_on[2], t_skip, p, q)
        for p, q in PQ_LIST
    ])


# -------------------------------------------------------------------------
# Grid sweep (parallel)
# -------------------------------------------------------------------------

def run_sweep(
    t_scale_vals : np.ndarray,
    J_inter_vals : np.ndarray,
    n_workers    : int = None,
    **kwargs
):
    """Compute uncoupled detuning (f1 - f2) and coupled PLV on a 2D grid.
    Simulations are dispatched across n_workers processes (default: all cores).
    PLV is returned per candidate (p, q) lock — shape (n_j, n_t, len(PQ_LIST))."""

    n_t = len(t_scale_vals)
    n_j = len(J_inter_vals)

    df_meas   = np.zeros(n_t)
    plv_vals  = np.zeros((n_j, n_t, len(PQ_LIST)))
    n_workers = n_workers or os.cpu_count()

    with ProcessPoolExecutor(max_workers=n_workers) as ex:

        # Uncoupled frequencies (one run per t_scale)
        freq_futs = {
            ex.submit(measure_freqs, tval, **kwargs): j
            for j, tval in enumerate(t_scale_vals)
        }
        for k, fut in enumerate(as_completed(freq_futs), 1):
            j          = freq_futs[fut]
            f1, f2     = fut.result()
            df_meas[j] = f1 - f2
            log = (
                f"[freq {k:2d}/{n_t}] "
                f"ts={t_scale_vals[j]:.3f}  "
                f"f1={f1:.3f}  f2={f2:.3f}  "
                f"df={df_meas[j]:+.3f}"
            )
            print(log)

        # Coupled PLV (full grid)
        plv_futs = {
            ex.submit(run_one, tval, jval, **kwargs): (i, j)
            for i, jval in enumerate(J_inter_vals)
            for j, tval in enumerate(t_scale_vals)
        }
        n_tot = len(plv_futs)
        for k, fut in enumerate(as_completed(plv_futs), 1):
            i, j              = plv_futs[fut]
            plv_vals[i, j, :] = fut.result()
            kbest             = int(np.nanargmax(plv_vals[i, j]))
            pbest, qbest      = PQ_LIST[kbest]
            log = (
                f"[plv  {k:3d}/{n_tot}] "
                f"J={J_inter_vals[i]:6.1f} pA  "
                f"ts={t_scale_vals[j]:.3f}  "
                f"best {pbest}:{qbest} PLV={plv_vals[i,j,kbest]:.3f}"
            )
            print(log)

    return df_meas, plv_vals


# -------------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------------

def plot_tongues(
    df_meas      : np.ndarray,
    J_inter_vals : np.ndarray,
    plv_vals     : np.ndarray,
    savepath     : str = None,
):
    """Arnold tongue heatmap: detuning vs coupling strength.
    Best p:q lock per grid point; tongues annotated by their ratio."""

    J_abs = np.abs(J_inter_vals)

    # Best (p, q) lock per grid point
    plv_max = np.nanmax(plv_vals, axis=2)
    pq_idx  = np.nanargmax(plv_vals, axis=2)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(
        df_meas, J_abs, plv_max,
        shading='auto', cmap='viridis', vmin=0.0, vmax=1.0,
    )
    cs = ax.contour(
        df_meas, J_abs, plv_max,
        levels=[0.5, 0.8, 0.95], colors='w', linewidths=1.0,
    )
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')

    # Annotate each strongly-locked region with its p:q ratio
    for k, (p, q) in enumerate(PQ_LIST):
        sub = np.where((pq_idx == k) & (plv_max > 0.8), plv_max, -1.0)
        if sub.max() < 0:
            continue
        i_m, j_m = np.unravel_index(np.argmax(sub), sub.shape)
        ax.text(
            df_meas[j_m], J_abs[i_m], f'{p}:{q}',
            color='w', fontsize=9, ha='center', va='center',
            fontweight='bold',
        )

    ax.axvline(0.0, color='k', lw=0.5, ls='--')
    ax.set_xlabel(r'Frequency detuning  $f_1 - f_2$  (Hz)')
    ax.set_ylabel(r'Inter-segment inhibition  $|J_{\mathrm{inter}}|$  (pA)')
    ax.set_title('Arnold tongues — 2-segment CPG phase locking')
    fig.colorbar(im, ax=ax, label='max PLV over p:q')
    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=140)
    return fig, ax


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def run_analysis():

    n_t_vals = 21       # ODD to include 1.0
    n_j_vals = 11

    # Time scales
    freq_scales  = np.linspace(0.1, 10.0, n_t_vals)
    t_scale_vals = np.sort(1.0 / freq_scales)

    # Weights
    J_max        = 100.0
    J_inter_vals = np.linspace(-0.0, -J_max, n_j_vals)

    # Run sweep
    df_meas, plv_vals = run_sweep(
        t_scale_vals,
        J_inter_vals,
        n_workers = None,
        duration  = 100.0,
        t_skip    = 10.0,
    )

    # Plot
    plot_tongues(
        df_meas      = df_meas,
        J_inter_vals = J_inter_vals,
        plv_vals     = plv_vals,
        savepath     = None,
    )
    plt.show()

if __name__ == "__main__":
    run_analysis()