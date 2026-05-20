"""Text blocks for the CPG neuronal and synaptic model equations.

This module keeps the publication-facing mathematical description separate
from plotting code. ``plot_results.py`` imports these helpers when composing
the architecture/equations figure.
"""

from __future__ import annotations


def neuron_model_text() -> str:
    """Return the adaptive integrate-and-fire neuron equations."""
    return (
        "Adaptive integrate-and-fire neuron\n"
        r"$\tau_{m,i}\frac{du_i}{dt}=-(u_i-E_{rest})-R_m w_i$" "\n"
        r"$\qquad + R_m\left[I_{ext,i}+\sum_j g_{ij}(E_{inh}-u_i)\right]$" "\n"
        r"$\tau_{w,i}\frac{dw_i}{dt}=-w_i$" "\n"
        r"Spike if $u_i\geq E_{th}$: $u_i\leftarrow E_{rest}$, "
        r"$w_i\leftarrow w_i+\Delta w$, refractory for $t_{ref,i}$."
    )


def synapse_model_text() -> str:
    """Return the inhibitory conductance synapse equations."""
    return (
        "Conductance-based inhibitory synapse\n"
        r"$\tau_{syn,i}\frac{dg_{ij}}{dt}=-g_{ij}$" "\n"
        r"After delay $d_{ij}$, a spike from neuron $j$ increments "
        r"$g_{ij}\leftarrow g_{ij}+W_{ij}$." "\n"
        r"$I_{syn,i}=\sum_j g_{ij}(E_{inh}-u_i)$"
    )


def model_equation_text() -> str:
    """Return neuron and synapse equations as a single formatted block."""
    return neuron_model_text() + "\n\n" + synapse_model_text()


def swept_parameters_text(
    *,
    f1: float,
    k1_abs: float,
    df_min: float,
    df_max: float,
    g_min: float,
    g_max: float,
    g_intra_nS: float,
) -> str:
    """Return a formatted description of the varied analysis parameters."""
    return (
        "Parameters changed in the analysis\n"
        r"1. Segment 1 is calibrated to the reference isolated frequency:" "\n"
        fr"   $f^{{iso}}_1\approx {f1:.2f}$ Hz, with absolute time scale $k_1={k1_abs:.3g}$." "\n"
        r"2. Segment 2 is detuned by a symmetric isolated-frequency mismatch:" "\n"
        fr"   $\Delta f=f^{{iso}}_2-f^{{iso}}_1\in[{df_min:.2f},{df_max:.2f}]$ Hz." "\n"
        r"   Positive $\Delta f$ means segment 2 is intrinsically faster." "\n"
        r"3. Detuning is implemented by segment-wise time dilation:" "\n"
        r"   $\tau_m,\tau_w,\tau_{syn},t_{ref},d \mapsto k_s(\cdot)$." "\n"
        r"4. Crossed inter-segment inhibition is swept on a log scale:" "\n"
        fr"   $g_{{inter}}={g_min:.4g}$–{g_max:.3g} nS; $g_{{intra}}={g_intra_nS:.0f}$ nS is fixed."
    )


def classification_text() -> str:
    """Return the synchronization classification criteria."""
    return (
        "Synchronization criterion\n"
        r"A point is classified as 1:1 synchronized only if both segments remain " "\n"
        r"oscillatory, their measured cycle frequencies match within tolerance, " "\n"
        r"the 1:1 phase-locking value is high, and the unwrapped phase difference " "\n"
        r"has negligible residual drift. All other valid oscillatory points are " "\n"
        r"classified as drift; non-1:1 relationships are not separated. The resulting " "\n"
        r"locked frequency is estimated directly from the slope of the simulated " "\n"
        r"common phase, not from the arithmetic mean of the two scalar frequencies."
    )


if __name__ == "__main__":
    print(model_equation_text())
