# VERSION: clean_phase_v3_2026_06_03
"""
Spiking CPG with AdEx neurons and minimal burst-based phase analysis.

The simulation is unchanged. The analysis is intentionally simple:
1. estimate one global ISI threshold from the largest gap in the log-ISI values;
2. split each spike train into bursts;
3. use burst starts to define a linear phase from 0 to 2*pi;
4. extend the final phase interval linearly to the end of the trace;
5. compare oscillators with the wrapped phase difference.
"""

from os import times

import numpy as np
import brian2 as b2
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
class SpikingCPG():

    def __init__(
        self,
        t_scale1,
        t_scale2,
        record   = False,
        callback = False,
    ):
        """Simulate the 4-neuron CPG."""

        self.t_scale1 = t_scale1
        self.t_scale2 = t_scale2

        self.record   = record
        self.callback = callback

        self.timestep = 0.1 * b2.ms
        self.duration = None
        self.n_steps  = None
        self.times    = None

        b2.defaultclock.dt      = self.timestep
        b2.prefs.codegen.target = 'cython'

        self.ner_params = dict(
            # AdEx neuron
            tau_m   = 16.7  * b2.ms,
            R_m     = 80    * b2.Mohm,
            V_rest  = -70   * b2.mV,
            V_rheo  = -50   * b2.mV,
            V_reset = -52   * b2.mV,
            V_thres = 0     * b2.mV,
            Delta_T = 2     * b2.mV,
            a_gain  = 2     * b2.nS,
            tau_w   = 180   * b2.ms,
            delta_w = 50    * b2.pA,
            I_ext   = 500   * b2.pA,
            t_refr  = 0     * b2.ms,

            # Synapses
            J_intra = -200  * b2.pA,
            J_inter = -50   * b2.pA,
            tau_s   = 30    * b2.ms,
            d_syn   = 0     * b2.ms,
        )

        # Define network
        self._define_neuron_group()
        self._define_synaptic_group()
        self._define_network()
        self._define_callback()

        # Output dictionary
        self.sim_out = None

        return

    # -------------------------------------------------------------------------
    # Network definition
    # -------------------------------------------------------------------------

    def _define_neuron_group(self):
        ''' Define the neuron group with AdEx dynamics  '''

        # Define equations
        ner_eqs = """
        t_scale                         : 1      (constant)
        tau_m_eff = tau_m  * t_scale    : second
        tau_w_eff = tau_w  * t_scale    : second
        tau_s_eff = tau_s  * t_scale    : second
        tau_refr  = t_refr * t_scale    : second

        v_lin = - (v - V_rest)                              : volt
        v_exp = + Delta_T * exp( (v - V_rheo) / Delta_T )   : volt
        dv/dt = ( v_lin + v_exp + R_m * I_tot ) / tau_m_eff : volt (unless refractory)

        I_tot = - w + I_ext + I_syn                         : amp
        dw/dt = ( a_gain * (v - V_rest) - w ) / tau_w_eff   : amp
        dI_syn/dt = - I_syn / tau_s_eff                     : amp
        """

        # Create the neuron group
        self.pop = b2.NeuronGroup(
            N          = 4,
            model      = ner_eqs,
            threshold  = "v >= V_thres",
            reset      = "v = V_reset; w += delta_w",
            refractory = "tau_refr",
            method     = "euler",
            namespace  = self.ner_params,
        )

        # Adjust time scales
        t1, t2 = self.t_scale1, self.t_scale2
        self.pop.t_scale = np.array([ t1, t1, t2, t2 ])

        # Initialize values
        self.pop.v = [
            self.ner_params["V_rheo"] - 1*b2.mV,
            self.ner_params["V_rest"] - 5*b2.mV,
            self.ner_params["V_rest"] - 5*b2.mV,
            self.ner_params["V_rheo"] - 1*b2.mV,
        ]
        self.pop.w     = 0 * b2.pA
        self.pop.I_syn = 0 * b2.pA

        return

    def _assign_synaptic_weights(
        self,
        J_intra : float = None,
        J_inter : float = None,
    ):
        ''' Assign synaptic weights. '''
        if J_intra is None:
            J_intra = self.ner_params["J_intra"]
        if J_inter is None:
            J_inter = self.ner_params["J_inter"]
        self.syn.J_syn = (
            [ - abs(J_intra) ] * self.n_intra_in +
            [ - abs(J_inter) ] * self.n_inter_in +
            [ + abs(J_inter) ] * self.n_inter_ex
        )

    def _define_synaptic_group(self):
        ''' Define the synaptic group  '''

        # Create synapses
        self.syn = b2.Synapses(
            source = self.pop,
            target = self.pop,
            model  = "J_syn : amp",
            on_pre = "I_syn_post += J_syn",
        )

        # Connect synapses
        self.wmat_intra_in = [(0, 1), (1, 0), (2, 3), (3, 2)]
        self.wmat_inter_in = [(0, 3), (3, 0), (1, 2), (2, 1)]

        # self.wmat_inter_ex = [(0, 2), (2, 0), (1, 3), (3, 1)]
        self.wmat_inter_ex = []

        self.wmat_all      = (
            self.wmat_intra_in +
            self.wmat_inter_in +
            self.wmat_inter_ex
        )

        self.syn.connect(
            i=[i for i, _ in self.wmat_all],
            j=[j for _, j in self.wmat_all],
        )

        # Define synaptic weights
        self.n_intra_in = len(self.wmat_intra_in)
        self.n_inter_in = len(self.wmat_inter_in)
        self.n_inter_ex = len(self.wmat_inter_ex)

        self._assign_synaptic_weights()

        # Define synaptic delay
        self.syn.delay = self.ner_params["d_syn"]

        return

    def _define_network(self):
        ''' Define brian network  '''

        self.spikemon = b2.SpikeMonitor(self.pop)
        self.network  = b2.Network(self.pop, self.syn, self.spikemon)

        if self.record:
            self.statemon = b2.StateMonitor(
                source    = self.pop,
                variables = ["v", "w", "I_syn"],
                record    = True,
                dt        = 1*b2.ms,
            )
            self.network.add(self.statemon)

        return

    # -------------------------------------------------------------------------
    # Callback function
    # -------------------------------------------------------------------------

    def _toggle_J_inter(self, curtime, t_toggle, active):

        if abs( float(curtime - t_toggle) ) > float( self.timestep / 2 ):
            return

        J_intra = self.ner_params["J_intra"]
        J_inter = self.ner_params["J_inter"] * int(active)
        J_state = 'ON' if active else 'OFF'

        print(f'Turning inter-segment connections {J_state}')

        self._assign_synaptic_weights(
            J_intra = J_intra,
            J_inter = J_inter,
        )

    def _callback_function(self, curtime):
        ''' Step function '''

        self._toggle_J_inter(curtime, 0 / 2 * self.duration, active= False)
        self._toggle_J_inter(curtime, 1 / 2 * self.duration, active= True )

        return

    def _define_callback(self):
        ''' Define brian callback '''

        if not self.callback:
            return

        self.netwop = b2.NetworkOperation(
            function = self._callback_function,
            name     = 'callback',
        )
        self.network.add(self.netwop)

        return

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------

    def run(self, duration):
        ''' Run the brian network and collect results  '''

        # Time parameters
        self.duration = duration
        self.n_steps  = int( duration / self.timestep )
        self.times    = np.arange(self.n_steps) * float(self.timestep)

        # Run the network
        self.network.run(duration)

        # Group spike trains
        spike_trains = self.spikemon.spike_trains()
        spike_trains = { i : np.array(st) for i, st in spike_trains.items() }

        # Collect results
        self.sim_out = {
            'times'       : self.times,
            "spike_trains": spike_trains,
            "params"      : self.ner_params,
        }
        if self.record:
            self.sim_out["t"]     = np.array( self.statemon.t     )
            self.sim_out["v"]     = np.array( self.statemon.v     )
            self.sim_out["w"]     = np.array( self.statemon.w     )
            self.sim_out["I_syn"] = np.array( self.statemon.I_syn )

        # Compute signals phases
        self.get_bursts()

        return self.sim_out

    # -------------------------------------------------------------------------
    # Phase analysis
    # -------------------------------------------------------------------------

    def _get_isi_threshold(self):
        """Get ISI threshold separating intra-burst and inter-burst."""

        # Get ISIs for all neurons
        isis = {
            i : np.sort( np.diff(st) )
            for i, st in self.sim_out["spike_trains"].items()
        }

        # Max jump in log-ISI
        splits = {
            i : np.argmax( np.diff( np.log(isis_i) ) )
            for i, isis_i in isis.items()
        }

        # Compute log-mean
        isi_th = {}
        for i in isis:
            isis_i    = isis[i]
            split_i   = splits[i]
            isi_th[i] = np.sqrt(isis_i[split_i] * isis_i[split_i + 1])

        return isi_th

    def _get_bursts_train(
        self,
        spike_train: np.ndarray,
        isi_th     : float,
    ):
        """Return the bursts of spikes."""
        spike_train   = np.asarray(spike_train, dtype=float)
        # Find candidates
        bursts_start  = np.where(np.diff(spike_train) > isi_th)[0] + 1
        # Split into trains
        bursts_train = np.split(spike_train, bursts_start)
        return bursts_train

    def _get_phase_from_burst_starts(
        self,
        burst_starts : np.ndarray,
    ):
        """Linear phase between burst starts, extended to the end of the trace."""

        two_pi = 2 * np.pi
        times  = self.times
        phases = np.full_like(times, 0.0, dtype=float)

        # Not enough bursts
        if len(burst_starts) < 2:
            return phases

        # Phase computation
        get_p = lambda i, t0, t1 : two_pi * (times[i] - t0) / (t1 - t0)

        # Phase evolution
        for tb_0, tb_1 in zip(burst_starts[:-1], burst_starts[1:]):
            b_idx        = (times >= tb_0) & (times < tb_1)
            phases[b_idx] = get_p(b_idx, tb_0, tb_1)

        # Handle first period
        first_idx         = times < burst_starts[0]
        phases[first_idx] = get_p(first_idx, burst_starts[0], burst_starts[1])

        # Handle last period
        last_idx         = times >= burst_starts[-1]
        phases[last_idx] = get_p(last_idx, burst_starts[-2], burst_starts[-1])

        return phases

    def _get_spike_rate(
        self,
        spike_train: np.ndarray,
        isi_th     : float,
    ):
        """Get instantaneous firing rate by binning spikes and Gaussian smoothing."""

        timestep     = float(self.timestep)                         # seconds
        spike_inds   = np.array(spike_train / timestep, dtype=int)
        spike_counts = np.zeros(self.n_steps)

        # Binning
        np.add.at(spike_counts, spike_inds, 1)

        # Smoothing
        spike_counts = gaussian_filter1d(
            spike_counts,
            sigma = 0.2 * isi_th / timestep,
            mode  = "constant",
        )

        # Convert to firing rate
        spike_rate = spike_counts / timestep

        return spike_rate

    def get_bursts(self):
        """Return phases, burst starts, and the global ISI threshold."""

        # Split spike trains into bursts
        isi_th = self._get_isi_threshold()

        # Study bursts
        bursts_trains = {
            i: self._get_bursts_train(st, isi_th[i])
            for i, st in self.sim_out["spike_trains"].items()
        }

        bursts_counts = {
            i: np.array( [len(b) for b in bst] )
            for i, bst in bursts_trains.items()
        }

        bursts_onsets = {
            i: np.array( [b[0] for b in bst] )
            for i, bst in bursts_trains.items()
        }

        bursts_phases = {
            i: self._get_phase_from_burst_starts(b_on)
            for i, b_on in bursts_onsets.items()
        }

        # Spike rates
        spike_rates = {
            i: self._get_spike_rate(st, isi_th[i])
            for i, st in self.sim_out["spike_trains"].items()
        }

        # Store in sim_out
        self.sim_out['isi_th']        = isi_th
        self.sim_out['bursts_trains'] = bursts_trains
        self.sim_out['bursts_counts'] = bursts_counts
        self.sim_out['bursts_onsets'] = bursts_onsets
        self.sim_out['bursts_phases'] = bursts_phases
        self.sim_out['spike_rates']   = spike_rates

        return

    # -------------------------------------------------------------------------
    # Utils
    # -------------------------------------------------------------------------

    def _get_inds_last_seconds(
        self,
        interval : float      = None,
        times    : np.ndarray = None,
    ):
        ''' Get indices of the last seconds of the simulation. '''
        if times is None:
            times = self.sim_out["times"]
        if interval is None:
            interval = float(self.duration)
        return times >= times[-1] - interval

    def _get_v_nullcline(self, v_m, I_add):
        """AdEx v-nullcline: value of w, for dv/dt = 0."""

        params  = self.sim_out["params"]
        R_m     = float( params["R_m"]     )
        V_rest  = float( params["V_rest"]  )
        V_rheo  = float( params["V_rheo"]  )
        Delta_T = float( params["Delta_T"] )
        I_ext   = float( params["I_ext"]   )

        v_term = -(v_m - V_rest) + Delta_T * np.exp((v_m - V_rheo) / Delta_T)
        v_null = v_term / R_m + I_ext + I_add

        return v_null

    def _get_w_nullcline(self, v_m):
        """AdEx w-nullcline: value of w, for dw/dt = 0."""
        params = self.sim_out["params"]
        a_gain = float( params["a_gain"] )
        V_rest = float( params["V_rest"] )
        return a_gain * (v_m - V_rest)

    def _get_neuron_styles(self):
        """Colors and line styles for neurons grouped by oscillator."""

        osc1_color = "#1f77b4"      # blue
        osc1_shade = "#005f99"      # darker blue, for N1

        osc2_color = "#ff7f0e"      # orange
        osc2_shade = "#b35a00"      # darker orange/brown, for N3

        return {
            0: {
                "color": osc1_color,
                "linestyle": "-",
                "label": "N0",
            },
            1: {
                "color": osc1_shade,
                "linestyle": "--",
                "label": "N1",
            },
            2: {
                "color": osc2_color,
                "linestyle": "-",
                "label": "N2",
            },
            3: {
                "color": osc2_shade,
                "linestyle": "--",
                "label": "N3",
            },
        }

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------

    def plot_isi_histogram(
        self,
    ):
        ''' Inter-spike-interval histogram '''

        styles = self._get_neuron_styles()
        isi_th = self.sim_out['isi_th']
        spikes = self.sim_out['spike_trains']
        isis   = {i: np.diff(st) for i, st in spikes.items()}

        fig, ax = plt.subplots(figsize=(5, 3))

        for i in isis:
            style = styles[i]
            ax.hist(
                isis[i],
                bins      = 30,
                alpha     = 0.5,
                edgecolor = "k",
                color     = style["color"],
                label     = style["label"]
            )
            ax.axvline(
                isi_th[i],
                linestyle = "--",
                linewidth = 1.5,
                color     = style["color"],
                label     = "thr"
            )

        ax.set_xlabel("Inter-spike interval (s)")
        ax.set_ylabel("Count")
        ax.set_title("ISI threshold for burst detection")
        ax.legend(loc='upper right')
        fig.tight_layout()
        return fig, ax

    def plot_raster(
        self,
        plot_time: float     = None,
        axis      : plt.Axes = None,
        decorate  : bool     = True,
    ):
        ''' Raster plot '''

        styles = self._get_neuron_styles()
        idx    = self._get_inds_last_seconds(plot_time)
        times  = self.sim_out['times'][idx]
        starts = self.sim_out['bursts_onsets']
        spikes = self.sim_out["spike_trains"]
        t0, t1 = times[0], times[-1]

        if axis is None:
            fig, ax = plt.subplots(figsize=(8, 3))
        else:
            fig, ax = axis.get_figure(), axis

        for neuron, spike_times in spikes.items():

            style       = styles[neuron]
            neuron_y    = neuron + 1.0
            spike_times = np.asarray(spike_times, dtype=float)
            inds_spikes = (spike_times >= t0) & (spike_times <= t1)
            inds_starts = (starts[neuron] >= t0) & (starts[neuron] <= t1)

            spikes_t = spike_times[inds_spikes]
            bursts_t = starts[neuron][inds_starts]
            bursts_y = np.full_like(bursts_t, neuron_y)

            ax.vlines(
                spikes_t,
                neuron_y - 0.4,
                neuron_y + 0.4,
                color = 'k',
            )
            ax.plot(
                bursts_t,
                bursts_y,
                marker          = "o",
                linestyle       = "None",
                color           = style["color"],
                label           = style["label"],
                markeredgecolor = "k",
                markeredgewidth = 0.5,
            )

        ax.set_xlim(t0, t1)
        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels(["N0", "N1", "N2", "N3"])
        ax.set_ylabel("Neuron")
        ax.invert_yaxis()

        if decorate:
            ax.set_xlabel("Time (s)")
            ax.set_title("Raster plot")
            ax.legend(loc='upper right')
            fig.tight_layout()

        return fig, ax

    def plot_phases(
        self,
        plot_time = None,
    ):
        ''' Plot phases and phase difference for the paper figure. '''

        # Unwrap phases for plotting
        times   = self.sim_out["times"]
        phases  = self.sim_out["bursts_phases"]
        rates   = self.sim_out['spike_rates']

        phase_0 = phases[0]
        phase_2 = phases[2]

        phase_0[np.isnan(phase_0)] = 0.0
        phase_2[np.isnan(phase_2)] = 0.0

        phase_0    = np.unwrap(phase_0)
        phase_2    = np.unwrap(phase_2)
        phase_diff = phase_2 - phase_0

        idx        = self._get_inds_last_seconds(plot_time)
        times      = times[idx]
        phase_0    = phase_0[idx]
        phase_2    = phase_2[idx]
        phase_diff = phase_diff[idx]

        t_00 = times[0]
        t_10 = times[-1]
        t_05 = ( t_10 + t_00 ) / 2

        max_rate = max( [r.max() for r in rates.values()] )

        # Figure
        fig, axes = plt.subplots(
            5, 1,
            figsize=(12, 10),
            sharex=True
        )
        styles = self._get_neuron_styles()
        vline  = lambda ax: ax.axvline(t_05, color='gray', linestyle='--')


        # ------------------------------------------------------------
        # Raster plot
        # ------------------------------------------------------------
        ax_ind = 0
        self.plot_raster(
            plot_time = times[-1],
            axis      = axes[ax_ind],
            decorate  = False,
        )
        vline(axes[ax_ind])

        # ------------------------------------------------------------
        # Oscillator 1 output (N0 and N1)
        # ------------------------------------------------------------
        ax_ind += 1
        axes[ax_ind].plot(
            times,
            rates[0],
            color     = styles[0]["color"],
            linestyle = styles[0]["linestyle"],
            label     = styles[0]["label"],
        )
        axes[ax_ind].plot(
            times,
            rates[1],
            color     = styles[1]["color"],
            linestyle = styles[1]["linestyle"],
            label     = styles[1]["label"],
        )
        vline(axes[ax_ind])
        axes[ax_ind].set_ylabel(r'Firing rate (Hz)')
        axes[ax_ind].set_ylim(0, 1.1 * max_rate)
        axes[ax_ind].legend(loc='upper right')

        # ------------------------------------------------------------
        # Oscillator 2 output (N2 and N3)
        # ------------------------------------------------------------
        ax_ind += 1
        axes[ax_ind].plot(
            times,
            rates[2],
            color     = styles[2]["color"],
            linestyle = styles[2]["linestyle"],
            label     = styles[2]["label"],
        )
        axes[ax_ind].plot(
            times,
            rates[3],
            color     = styles[3]["color"],
            linestyle = styles[3]["linestyle"],
            label     = styles[3]["label"],
        )
        vline(axes[ax_ind])
        axes[ax_ind].set_ylabel(r'Firing rate (Hz)')
        axes[ax_ind].set_ylim(0, 1.1 * max_rate)
        axes[ax_ind].legend(loc='upper right')

        # ------------------------------------------------------------
        # Phase difference
        # ------------------------------------------------------------
        ax_ind += 1
        axes[ax_ind].plot(
            times,
            phase_diff,
            color     = 'black',
            linestyle = '-',
            label     = r'$\theta_2 - \theta_0$',
        )
        vline(axes[ax_ind])
        axes[ax_ind].set_ylabel(r'$\theta_2 - \theta_0$')

        # ------------------------------------------------------------
        # Phase evolution
        # ------------------------------------------------------------
        ax_ind += 1
        axes[ax_ind].plot(
            times,
            phase_0,
            color     = styles[0]["color"],
            linestyle = styles[0]["linestyle"],
            label     = r'$\theta_0$',
        )
        axes[ax_ind].plot(
            times,
            phase_2,
            color     = styles[2]["color"],
            linestyle = styles[2]["linestyle"],
            label     = r'$\theta_2$',
        )
        vline(axes[ax_ind])
        axes[ax_ind].set_ylabel(r'$\theta_0,\theta_2$')
        axes[ax_ind].set_xlabel('Time [s]')
        axes[ax_ind].legend(loc='upper right')

        # Mark the coupling activation time
        #for ax in axes:
        #    ax.grid(True, alpha=0.3)
        #    ax.axvspan(t_00, t_05, alpha=0.08)
        #    ax.axvspan(t_05, t_10, alpha=0.08)

        plt.tight_layout()
        return

    def plot_voltage(
        self,
        plot_time: float = None,
    ):
        ''' Plot voltage evolution of neurons '''

        styles = self._get_neuron_styles()
        idx    = self._get_inds_last_seconds(plot_time, self.sim_out["t"])
        times  = self.sim_out["t"][idx]
        v_m    = self.sim_out['v'][:, idx] * 1000.0

        # Get indices for the last seconds

        fig, axes = plt.subplots(4, 1, figsize=(8, 6), sharex=True)
        for i, ax in enumerate(axes):
            style = styles[i]
            ax.plot(
                times,
                v_m[i],
                color     = style["color"],
                label     = style["label"],
                linestyle = '-',
            )
            ax.set_ylabel("v (mV)")
            ax.legend(loc='upper right')

        axes[-1].set_xlabel("Time (s)")
        axes[0].set_title("Membrane potentials")
        fig.tight_layout()
        return fig, axes

    def plot_phase_plane(
        self,
        neuron   : int = 0,
        plot_time: float = None,
    ):
        ''' Phase plane evolution of one neuron '''

        idx    = self._get_inds_last_seconds(plot_time, self.sim_out["t"])
        v      = self.sim_out["v"][neuron, idx]
        w      = self.sim_out["w"][neuron, idx]
        I_syn  = self.sim_out["I_syn"][neuron, idx]
        params = self.sim_out["params"]

        v_tol = +5.0  * 1e-3
        v_low = -90.0 * 1e-3
        v_min = min( v.min() - v_tol, v_low )
        v_max = float( params["V_rheo"] ) + v_tol

        I_syn_0 = 0.0
        I_syn_1 = float( np.mean(I_syn) )

        v_grid     = np.linspace(v_min, v_max, 400)
        v_null_0   = self._get_v_nullcline(v_grid, I_syn_0)
        v_null_syn = self._get_v_nullcline(v_grid, I_syn_1)
        w_null     = self._get_w_nullcline(v_grid)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(v, w, lw=0.7)
        ax.plot(v_grid, v_null_0, lw=1.5, label="v-null 0")
        ax.plot(v_grid, v_null_syn, linestyle="--", lw=1.5, label="v-null <Isyn>")
        ax.plot(v_grid, w_null, lw=1.5, label="w-null")

        ax.set_xlabel("Membrane potential v (mV)")
        ax.set_ylabel("Adaptation current w (pA)")
        ax.set_title(f"Phase plane: N{neuron}")
        ax.legend(loc='upper right')
        fig.tight_layout()
        return fig, ax

    def plots(
        self,
        plot_time : float = None
    ):
        ''' Plots '''

        self.plot_isi_histogram()
        self.plot_raster(plot_time=plot_time)
        self.plot_phases(plot_time=plot_time)

        if self.record:
            self.plot_voltage(plot_time=plot_time)
            self.plot_phase_plane(neuron=0, plot_time=plot_time)

        plt.show()

# -------------------------------------------------------------------------
# Example
# -------------------------------------------------------------------------

def run_test():

    duration = 5.0 * b2.second
    t_scale1 = 1.5
    t_scale2 = 1.0

    # Run simulation
    cpg_sim = SpikingCPG(
        t_scale1 = t_scale1,
        t_scale2 = t_scale2,
        record   = True,
        callback = True,
    )
    cpg_sim.run(duration)

    # Plot results
    cpg_sim.plots()

    return

if __name__ == "__main__":
    run_test()
