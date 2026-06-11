"""
Simple sweep of t_scale2 vs CPG frequency and spikes per burst.

Put this file next to cpg_sim_ad_exp.py and run:

    python cpg_time_scaling_sweep_simple.py

Theory for the clean uncoupled test, J_INTER = 0:
    segment 1: f(t_scale2) ~= constant
    segment 2: f(t_scale2) ~= f(1) / t_scale2
    spikes per burst ~= constant
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import brian2 as b2
import matplotlib.pyplot as plt

from cpg_simulation import SpikingCPG

# -------------------------------------------------------------------------
# Analysis
# -------------------------------------------------------------------------

def run_one(
    t_scale : float,
    duration: float = 50.0,
    t_skip  : float = 10.0,
):

    # Build CPG
    cpg_sim = SpikingCPG(
        t_scale1 = t_scale,
        t_scale2 = t_scale,
        record   = False,
        callback = False,
    )
    cpg_sim._assign_synaptic_weights(J_inter = 0.0 * b2.pA)

    # Run CPG
    cpg_out = cpg_sim.run(duration * b2.second)

    # Study bursts
    bursts_trains = cpg_out["bursts_trains"]
    bursts_counts = cpg_out["bursts_counts"]
    bursts_onsets = cpg_out["bursts_onsets"]

    periods = []
    counts  = []

    for ner_ind in range(4):

        ner_b_train = bursts_trains[ner_ind]
        ner_b_count = bursts_counts[ner_ind]
        ner_b_onset = bursts_onsets[ner_ind]

        # Remove transient
        ok_bursts = np.where(ner_b_onset >= t_skip)[0]

        ner_b_train = [ ner_b_train[i] for i in ok_bursts[:-1] ]
        ner_b_count = [ ner_b_count[i] for i in ok_bursts[:-1] ]
        ner_b_onset = [ ner_b_onset[i] for i in ok_bursts[:-1] ]

        # Collect results
        periods_list = list( np.diff(ner_b_onset) )
        counts_list  = list(ner_b_count)

        periods.extend(periods_list)
        counts.extend(counts_list)

    bursts_properties = {
        'frequency'       : 1 / np.mean(periods),
        'spikes_per_burst': np.mean(counts),
    }

    return bursts_properties

# -------------------------------------------------------------------------
# Grid sweep (parallel)
# -------------------------------------------------------------------------

def run_sweep(
    t_scale_vals : np.ndarray,
    n_workers    : int = None,
):
    """Sweep t_scale and measure frequency and spikes per burst."""

    n_workers        = n_workers or os.cpu_count()
    n_runs           = len(t_scale_vals)

    sweep_results = {
        "frequencies"      : np.zeros(n_runs),
        "spikes_per_burst" : np.zeros(n_runs),
    }

    with ProcessPoolExecutor(max_workers=n_workers) as ex:

        run_results = {
            ex.submit(run_one, t_scale): run_ind
            for run_ind, t_scale in enumerate(t_scale_vals)
        }

        for k, run_id in enumerate(as_completed(run_results), 1):

            run_ind = run_results[run_id]
            result  = run_id.result()

            frequency        = result["frequency"]
            spikes_per_burst = result["spikes_per_burst"]

            sweep_results["frequencies"][run_ind]      = frequency
            sweep_results["spikes_per_burst"][run_ind] = spikes_per_burst

            print(
                f"[run {k:3d}/{n_runs}] "
                f"ts={t_scale_vals[run_ind]:.3f}  "
                f"f={frequency:.3f} Hz  "
                f"spikes/burst={spikes_per_burst:.2f}"
            )

    return sweep_results

# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def plot_results(
    t_scales     : np.ndarray,
    sweep_results: dict[str, np.ndarray],
):
    """Compare measured frequency and spikes/burst with theoretical scaling."""

    frequencies      = sweep_results["frequencies"]
    spikes_per_burst = sweep_results["spikes_per_burst"]

    # Reference value at t_scale closest to 1
    ref_ind = np.argmin(np.abs(t_scales - 1.0))

    f_ref      = frequencies[ref_ind]
    spike_ref  = spikes_per_burst[ref_ind]

    # Theoretical expectations
    frequencies_theory = f_ref / t_scales
    spikes_theory      = np.full_like(t_scales, spike_ref)

    # Errors
    frequency_error = frequencies - frequencies_theory
    frequency_error_percent = 100 * frequency_error / frequencies_theory

    spike_error = spikes_per_burst - spikes_theory

    # ---------------------------------------------------------------------
    # Frequency: measured vs theoretical
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        t_scales,
        frequencies,
        "o-",
        label="measured",
    )
    ax.plot(
        t_scales,
        frequencies_theory,
        "--",
        label="theory: f(1) / t_scale",
    )

    ax.set_xlabel("time scale")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title("Frequency vs time scale")
    ax.legend()
    fig.tight_layout()

    # ---------------------------------------------------------------------
    # Frequency error
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.plot(
        t_scales,
        frequency_error_percent,
        "o-",
    )

    ax.set_xlabel("time scale")
    ax.set_ylabel("frequency error (%)")
    ax.set_title("Frequency deviation from theory")
    fig.tight_layout()

    # ---------------------------------------------------------------------
    # Spikes per burst: measured vs theoretical
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        t_scales,
        spikes_per_burst,
        "o-",
        label="measured",
    )
    ax.plot(
        t_scales,
        spikes_theory,
        "--",
        label="theory: constant spikes/burst",
    )

    ax.set_xlabel("time scale")
    ax.set_ylabel("spikes per burst")
    ax.set_title("Spikes per burst vs time scale")
    ax.legend()
    fig.tight_layout()

    # ---------------------------------------------------------------------
    # Spikes per burst error
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.plot(
        t_scales,
        spike_error,
        "o-",
    )

    ax.set_xlabel("time scale")
    ax.set_ylabel("spikes per burst error")
    ax.set_title("Spike-count deviation from theory")
    fig.tight_layout()

    plt.show()

    return

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def run_analysis():

    freq_scales = np.linspace(0.1, 10.0, 21)
    t_scales    = np.sort(1.0 / freq_scales)

    if not np.any(np.isclose(t_scales, 1.0)):
        t_scales = np.sort(np.append(t_scales, 1.0))

    # Run sweep
    sweep_results = run_sweep(t_scales)

    # Plots
    plot_results(t_scales, sweep_results)

    return



if __name__ == "__main__":
    run_analysis()
