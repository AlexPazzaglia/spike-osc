"""
Spiking CPG simulator for two coupled half-center oscillators.

Architecture
------------
Segment 1: neurons 0 and 1, mutually inhibitory.
Segment 2: neurons 2 and 3, mutually inhibitory.
Inter-segment coupling is crossed inhibition: 0 <-> 3 and 1 <-> 2.

Frequency control
-----------------
Each segment has a time-scale factor k. For segment i, tau_memb, tau_w,
tau_syn for incoming synapses, refractory time, and optionally synaptic delay
are multiplied by k. This approximately implements time dilation of the
isolated segment and is much more robust than changing a single parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - fallback for environments without numba
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def wrap(func):
            return func
        if args and callable(args[0]):
            return args[0]
        return wrap


@dataclass(frozen=True)
class CPGParams:
    # Neuron model: adaptive integrate-and-fire
    tau_memb: float = 24.8e-3       # s
    tau_w: float = 210.8e-3         # s
    E_rest: float = -70e-3          # V
    E_th: float = -55e-3            # V
    R_memb: float = 100e6           # Ohm
    Delta_w: float = 25e-12         # A, adaptation increment per spike
    t_refr: float = 2.48e-3         # s
    I_ext: float = 240e-12          # A

    # Inhibitory synapse
    tau_syn: float = 12.4e-3        # s
    E_inh: float = -80e-3           # V
    delay: float = 2.48e-3          # s

    # Coupling defaults
    g_intra: float = 50e-9          # S
    g_inter: float = 20e-9          # S

    # Numerical integration
    dt: float = 0.1e-3              # s


def params_from_dict(overrides: Optional[Dict[str, float]] = None) -> CPGParams:
    """Return CPGParams, optionally replacing fields by a dict."""
    base = asdict(CPGParams())
    if overrides:
        unknown = set(overrides) - set(base)
        if unknown:
            raise ValueError(f"Unknown parameter(s): {sorted(unknown)}")
        base.update(overrides)
    return CPGParams(**base)


def build_W(g_intra: float, g_inter: float) -> np.ndarray:
    """Connectivity matrix W[post, pre]."""
    W = np.zeros((4, 4), dtype=np.float64)
    # Intra-segment mutual inhibition
    W[1, 0] = g_intra
    W[0, 1] = g_intra
    W[3, 2] = g_intra
    W[2, 3] = g_intra
    # Crossed inter-segment inhibition
    W[3, 0] = g_inter
    W[0, 3] = g_inter
    W[2, 1] = g_inter
    W[1, 2] = g_inter
    return W


def _segment_scale_vector(k1: float, k2: float) -> np.ndarray:
    return np.array([k1, k1, k2, k2], dtype=np.float64)


def _delay_steps_matrix(params: CPGParams, k1: float, k2: float,
                        scale_delay: bool) -> np.ndarray:
    """Return delay steps per connection [post, pre].

    With scale_delay=True, delay is scaled by the post-synaptic segment's
    time-scale. This makes isolated segment dynamics closer to exact time
    dilation. With scale_delay=False, delays are fixed in absolute time.
    """
    scales = _segment_scale_vector(k1, k2)
    out = np.zeros((4, 4), dtype=np.int64)
    for post in range(4):
        k = scales[post] if scale_delay else 1.0
        out[post, :] = max(1, int(round(params.delay * k / params.dt)))
    return out


@njit(cache=True, fastmath=True)
def _run_core(
    n_steps: int,
    dt: float,
    tau_memb_vec: np.ndarray,
    tau_w_vec: np.ndarray,
    refr_steps_vec: np.ndarray,
    tau_syn_vec: np.ndarray,
    E_rest: float,
    E_th: float,
    R_memb: float,
    Delta_w: float,
    E_inh: float,
    I_ext_vec: np.ndarray,
    W: np.ndarray,
    delay_steps: np.ndarray,
    u0: np.ndarray,
    max_spikes_per_neuron: int,
    record_stride: int,
):
    N = 4
    u = u0.copy()
    w_adapt = np.zeros(N)
    g_syn = np.zeros((N, N))
    refr_cnt = np.zeros(N, dtype=np.int64)

    max_delay = 1
    for i in range(N):
        for j in range(N):
            if delay_steps[i, j] > max_delay:
                max_delay = delay_steps[i, j]
    buf_size = max_delay + 2
    pending = np.zeros((buf_size, N, N))

    spk_t = np.zeros((N, max_spikes_per_neuron))
    spk_n = np.zeros(N, dtype=np.int64)
    spk_overflow = np.zeros(N, dtype=np.int64)

    if record_stride <= 0:
        record_stride = n_steps + 1
    n_rec = n_steps // record_stride + 1
    t_rec = np.zeros(n_rec)
    u_rec = np.zeros((n_rec, N))
    rec_i = 0

    for step in range(n_steps):
        # Deliver synaptic events scheduled for this step.
        idx = step % buf_size
        for post in range(N):
            for pre in range(N):
                g_syn[post, pre] += pending[idx, post, pre]
                pending[idx, post, pre] = 0.0

        # Inhibitory synaptic current and voltage update.
        for i in range(N):
            g_tot = 0.0
            for j in range(N):
                g_tot += g_syn[i, j]
            I_syn_i = (E_inh - u[i]) * g_tot

            if refr_cnt[i] == 0:
                u[i] += (dt / tau_memb_vec[i]) * (
                    -(u[i] - E_rest)
                    - R_memb * w_adapt[i]
                    + R_memb * (I_syn_i + I_ext_vec[i])
                )

        # Adaptation decay.
        for i in range(N):
            w_adapt[i] += (dt / tau_w_vec[i]) * (-w_adapt[i])

        # Synaptic decay. tau_syn_vec is per post-synaptic neuron.
        for post in range(N):
            decay = dt / tau_syn_vec[post]
            for pre in range(N):
                g_syn[post, pre] += decay * (-g_syn[post, pre])

        # Refractory countdown.
        for i in range(N):
            if refr_cnt[i] > 0:
                refr_cnt[i] -= 1

        # Threshold crossing and delayed event scheduling.
        for pre in range(N):
            if refr_cnt[pre] == 0 and u[pre] >= E_th:
                if spk_n[pre] < max_spikes_per_neuron:
                    spk_t[pre, spk_n[pre]] = step * dt
                    spk_n[pre] += 1
                else:
                    spk_overflow[pre] += 1

                for post in range(N):
                    if W[post, pre] != 0.0:
                        dst = (step + delay_steps[post, pre]) % buf_size
                        pending[dst, post, pre] += W[post, pre]

                u[pre] = E_rest
                w_adapt[pre] += Delta_w
                refr_cnt[pre] = refr_steps_vec[pre]

        if step % record_stride == 0:
            if rec_i < n_rec:
                t_rec[rec_i] = step * dt
                for i in range(N):
                    u_rec[rec_i, i] = u[i]
                rec_i += 1

    return spk_t, spk_n, spk_overflow, t_rec[:rec_i], u_rec[:rec_i]


def run_sim(
    T: float = 20.0,
    k1: float = 1.0,
    k2: float = 1.0,
    g_intra: Optional[float] = None,
    g_inter: Optional[float] = None,
    params: Optional[Union[CPGParams, Dict[str, float]]] = None,
    u0: Optional[np.ndarray] = None,
    scale_delay: bool = True,
    record: bool = False,
    record_dt: float = 1e-3,
    max_spike_rate_hz: float = 500.0,
):
    """Run the 4-neuron CPG simulation.

    Returns
    -------
    If record=False:
        list[np.ndarray], one spike-time array per neuron.
    If record=True:
        dict with keys: spikes, t, u, spike_overflow.
    """
    p = params if isinstance(params, CPGParams) else params_from_dict(params)
    if g_intra is None:
        g_intra = p.g_intra
    if g_inter is None:
        g_inter = p.g_inter

    dt = p.dt
    n_steps = int(round(T / dt))

    scales = _segment_scale_vector(k1, k2)
    tau_memb_vec = p.tau_memb * scales
    tau_w_vec = p.tau_w * scales
    tau_syn_vec = p.tau_syn * scales
    refr_steps_vec = np.maximum(1, np.round(p.t_refr * scales / dt)).astype(np.int64)

    I_ext_vec = np.full(4, p.I_ext, dtype=np.float64)
    W = build_W(g_intra, g_inter)
    delay_steps = _delay_steps_matrix(p, k1, k2, scale_delay=scale_delay)

    if u0 is None:
        u0 = np.array([
            p.E_th - 1e-3,
            p.E_rest - 8e-3,
            p.E_rest - 8e-3,
            p.E_th - 1e-3,
        ], dtype=np.float64)
    else:
        u0 = np.asarray(u0, dtype=np.float64)
        if u0.shape != (4,):
            raise ValueError("u0 must have shape (4,)")

    max_spikes_per_neuron = max(100, int(np.ceil(T * max_spike_rate_hz)))
    record_stride = max(1, int(round(record_dt / dt))) if record else n_steps + 1

    spk_t, spk_n, spk_overflow, t_rec, u_rec = _run_core(
        n_steps,
        dt,
        tau_memb_vec.astype(np.float64),
        tau_w_vec.astype(np.float64),
        refr_steps_vec,
        tau_syn_vec.astype(np.float64),
        p.E_rest,
        p.E_th,
        p.R_memb,
        p.Delta_w,
        p.E_inh,
        I_ext_vec,
        W.astype(np.float64),
        delay_steps,
        u0,
        max_spikes_per_neuron,
        record_stride,
    )

    spikes = [spk_t[i, :spk_n[i]].copy() for i in range(4)]
    if not record:
        return spikes
    return {
        "spikes": spikes,
        "t": t_rec.copy(),
        "u": u_rec.copy(),
        "spike_overflow": spk_overflow.copy(),
    }


def split_into_bursts(spike_times: np.ndarray, isi_gap: float = 0.15) -> List[np.ndarray]:
    """Split spike times into bursts using an inter-spike gap threshold."""
    t = np.asarray(spike_times, dtype=float)
    if t.size == 0:
        return []
    if t.size == 1:
        return [t]
    cut = np.where(np.diff(t) > isi_gap)[0]
    starts = np.r_[0, cut + 1]
    ends = np.r_[cut + 1, t.size]
    return [t[s:e] for s, e in zip(starts, ends)]
