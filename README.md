# Spiking CPG synchronization regime-map analysis

This package simulates two coupled spiking half-center oscillators and maps their synchronization behavior as a function of two experimentally controlled quantities:

1. the isolated frequency mismatch between the two oscillators,
2. the crossed inter-segment inhibitory coupling strength.

The output is an empirical synchronization regime map with an Arnold-tongue-like 1:1 locking region. The system is not reduced to a closed-form phase-oscillator model; therefore drift, n:m locking, irregular activity, asymmetric dominance, and loss of valid oscillation are explicitly tracked rather than hidden.

## Version 4 changes

This version adds four publication-oriented improvements:

- Representative traces now display a short default window of 2 s, so individual spikes and alternation are visible. The simulation itself still uses the full duration saved in the sweep file; only the displayed interval is cropped.
- Log files are written inside the relevant results directory. The smoke-test log is saved as `smoke_results/run_smoke_tests.log`. The package root is kept free of generated logs.
- `plot_results.py` now produces a network/methods schematic: `network_architecture_equations.png/.pdf`. It shows the four-neuron architecture, fixed intra-segment inhibition, swept inter-segment inhibition, the neuron/synapse equations, and the parameters varied in the sweep.
- The README now contains a methods-level description that can be adapted later for a scientific publication.

## Files

- `cpg_sim.py` — simulator for the four-neuron adaptive integrate-and-fire CPG.
- `analysis.py` — burst detection, oscillator validity criteria, phase reconstruction, phase-locking metrics, and regime classification.
- `sweep_arnold.py` — calibration of the reference oscillator, construction of the symmetric detuning/coupling grid, coupled sweep, and data export.
- `plot_results.py` — publication-oriented plots: main regime map, representative traces, and architecture/equation schematic.
- `summarize_results.py` — regime counts, empirical 1:1 locking boundary, locking width summaries, and interpretation JSON.
- `run_smoke_tests.py` — end-to-end validation on a reduced grid.
- `requirements.txt` — dependencies.

## Installation

```bash
pip install -r requirements.txt
```

`numba` is recommended. Without it, the simulation will be much slower.

## Quick validation

```bash
python run_smoke_tests.py
```

This creates all validation outputs in:

```text
smoke_results/
```

including:

```text
smoke_results/run_smoke_tests.log
smoke_results/sweep_data.npz
smoke_results/arnold_main.png
smoke_results/example_traces.png
smoke_results/network_architecture_equations.png
smoke_results/interpretation_summary.json
```

## Full default analysis

```bash
python sweep_arnold.py --out results
python plot_results.py --data results/sweep_data.npz
python summarize_results.py --data results/sweep_data.npz
```

Default settings:

- target reference frequency: `3.0 Hz`
- symmetric target mismatch range: approximately `[-2.4, +2.4] Hz`
- detuning samples: `61`
- coupling samples: `45`, including exact zero coupling
- smallest nonzero coupling: `0.002 nS`
- maximum coupling: `0.5 * g_intra`
- automatic duration adjustment: enabled unless `--T` is explicitly provided
- representative trace display window: `2 s`

## Wide analysis

```bash
python sweep_arnold.py --out results_wide --delta-f-max-Hz 2.7 --n-delta 81 --n-g 50
python plot_results.py --data results_wide/sweep_data.npz
python summarize_results.py --data results_wide/sweep_data.npz
```

For the wide command, the slow edge has a segment-2 target frequency of approximately 0.3 Hz when the reference oscillator is near 3 Hz. The sweep script therefore automatically increases the simulation duration when `--T` is not set, so the slowest oscillator still contributes enough post-transient cycles for classification.

To change the displayed trace window:

```bash
python plot_results.py --data results_wide/sweep_data.npz --example-window-s 2.0
```

## Keeping logs inside result folders

No script intentionally writes log files to the package root. Recommended command-line logging should redirect output into the selected results directory:

```bash
python sweep_arnold.py --out results_wide --delta-f-max-Hz 2.7 --n-delta 81 --n-g 50 > results_wide/run.log 2>&1
python plot_results.py --data results_wide/sweep_data.npz >> results_wide/run.log 2>&1
python summarize_results.py --data results_wide/sweep_data.npz >> results_wide/run.log 2>&1
```

## Scientific methods description

### Network architecture

The model consists of two coupled spiking central-pattern-generator-like segments. Each segment is a half-center oscillator composed of two adaptive integrate-and-fire neurons connected by reciprocal inhibition. Segment 1 contains neurons N0 and N1; segment 2 contains neurons N2 and N3. Within each segment, the two neurons inhibit each other through a fixed intra-segment conductance `g_intra`. The two segments are coupled through crossed inhibitory projections: N0 inhibits N3 and N3 inhibits N0, while N1 inhibits N2 and N2 inhibits N1. The crossed inter-segment conductance `g_inter` is the coupling parameter swept in the analysis.

The architecture therefore separates two forms of inhibition:

- `g_intra`: fixed, strong, reciprocal inhibition within each half-center oscillator;
- `g_inter`: swept, weaker, crossed inhibition between the two oscillators.

By default, `g_inter` is sampled logarithmically from a very small nonzero value to at most `0.5 * g_intra`, with an additional exact zero-coupling row. This prevents the analysis from being dominated by unrealistically strong inter-segment coupling values close to the intra-segment weight, while preserving high resolution near the weak-coupling tongue tip.

### Neuron model

Each neuron is represented by an adaptive integrate-and-fire model. The membrane voltage `u_i` and adaptation current `w_i` evolve according to:

```math
\tau_{m,i}\frac{du_i}{dt}
= -(u_i-E_{rest}) - R_m w_i
+ R_m\left[I_{ext,i}+\sum_j g_{ij}(E_{inh}-u_i)\right]
```

```math
\tau_{w,i}\frac{dw_i}{dt}=-w_i
```

When the membrane voltage crosses threshold,

```math
u_i \geq E_{th},
```

a spike is registered, the voltage is reset to `E_rest`, the adaptation current is incremented by `Delta_w`, and the neuron is held refractory for `t_ref,i`:

```math
u_i \leftarrow E_{rest}, \qquad w_i \leftarrow w_i + \Delta w.
```

The external current `I_ext`, adaptation increment `Delta_w`, membrane resistance, voltage thresholds, and reversal potentials are held fixed during a sweep unless explicitly changed from the command line.

### Synapse model

Inhibitory synapses are conductance-based. For a synapse from presynaptic neuron `j` to postsynaptic neuron `i`, the conductance `g_ij` decays exponentially:

```math
\tau_{syn,i}\frac{dg_{ij}}{dt}=-g_{ij}.
```

When a presynaptic spike arrives after the configured delay, the conductance is incremented by the corresponding connectivity-matrix element:

```math
g_{ij} \leftarrow g_{ij}+W_{ij}.
```

The synaptic current contribution to neuron `i` is:

```math
I_{syn,i}=\sum_j g_{ij}(E_{inh}-u_i).
```

Because `E_inh` is below the typical membrane voltage, this term is hyperpolarizing.

### Frequency manipulation by time dilation

The frequency mismatch is imposed by segment-wise time dilation rather than by changing a single biophysical parameter. For a segment `s` with time-scale factor `k_s`, the following quantities are scaled together:

```math
\tau_m,\ \tau_w,\ \tau_{syn},\ t_{ref},\ d \mapsto k_s(\tau_m,\tau_w,\tau_{syn},t_{ref},d).
```

This is intentionally conservative. Changing only one parameter, such as `tau_w` or `I_ext`, can change the rhythm-generating mechanism itself and may push the oscillator into tonic or silent regimes. Scaling the relevant time constants together approximately preserves the isolated oscillator's trajectory in state space while changing the speed at which the trajectory is traversed. Under ideal time dilation, the frequency scales approximately as:

```math
f(k) \approx \frac{f(k=1)}{k}.
```

The analysis does not rely blindly on this approximation. It uses the approximation only to construct the target sweep grid, then measures the isolated frequency of each oscillator directly before plotting or classifying the coupled system.

### Reference-frequency calibration

The script first estimates the time-scale factor needed for segment 1 to oscillate near the requested reference frequency, defaulting to 3 Hz. It does this by measuring the isolated frequency at `k=1`, then estimating:

```math
k_1 \approx \frac{f(k=1)}{f^{target}_1}.
```

The estimated `k1` is then verified by another isolated simulation. If the resulting segment is not classified as oscillatory, the script raises an error. This prevents a coupled sweep from being run around an invalid reference oscillator.

### Symmetric frequency-mismatch grid

After reference calibration, the sweep is defined in terms of the target isolated mismatch:

```math
\Delta f = f^{iso}_1 - f^{iso}_2.
```

Positive `Delta f` means segment 2 is slower than segment 1. Negative `Delta f` means segment 2 is faster. The grid is symmetric by construction:

```math
\Delta f \in [-\Delta f_{max}, +\Delta f_{max}].
```

The target segment-2 frequency is:

```math
f^{target}_2 = f^{iso}_1 - \Delta f.
```

This is converted into a segment-2 time scale using the time-dilation approximation. Then the actual isolated segment-2 frequency is measured for every point of the grid. All final plots use measured `Delta f`, not only nominal target `Delta f`.

The slow side of the grid is constrained by a minimum allowed target segment-2 frequency, controlled by:

```bash
--min-f2-Hz
```

By default this is `0.25 Hz`. If a requested `Delta f_max` would make segment 2 slower than this floor, the script clips the requested range unless `--strict-delta-limit` is passed.

### Automatic simulation duration

Wide detuning ranges can include very slow oscillators. A fixed simulation duration can then become a classification artifact: the slow oscillator may generate too few post-transient cycles even though it would oscillate if simulated longer. To avoid this, the script estimates a recommended duration from the slowest target frequency and the largest segment time scale:

```math
T_{recommended} \approx T_{transient} + \frac{N_{cycles}}{f_{min}} + T_{buffer}.
```

The default minimum number of post-transient cycles is controlled by:

```bash
--min-eval-cycles
```

When `--T` is not explicitly specified, automatic duration adjustment is enabled by default. If the requested/default duration is too short, the script increases `T` and records the final value in `summary.json` and `sweep_data.npz`.

### Isolated oscillator validity

Before coupled dynamics are interpreted, the isolated oscillators are tested. Spike trains are split into bursts using an inter-spike gap threshold scaled by the segment time scale. A segment is considered a valid oscillator only if both neurons contribute enough bursts, the burst-count balance is sufficient, the alternation fraction is high, and the reference inter-burst interval is regular.

The default thresholds in `analysis.py` are:

- minimum cycles per neuron: `MIN_CYCLES = 4`
- maximum inter-burst-interval coefficient of variation: `MAX_CV = 0.35`
- minimum alternation fraction: `MIN_ALT_FRAC = 0.70`
- minimum burst-count balance between the two neurons: `MIN_BALANCE = 0.35`

The possible segment statuses are:

- `osc`: valid alternating oscillator;
- `silent`: too few spikes to characterize;
- `asymmetric`: one neuron dominates or the two neurons do not both burst;
- `tonic`: sustained firing without clean alternation;
- `irregular`: bursts exist but their timing is too variable.

### Coupled sweep and regime classification

For each pair of measured frequency mismatch and inter-segment coupling, the model is simulated with the two segments coupled by crossed inhibition. The two segments are first analyzed independently using the isolated-oscillator criteria. If either segment fails to remain a valid oscillator, the grid point is classified as `no_osc`, and the corresponding reason is saved.

If both segments remain oscillatory, their cycle frequencies are estimated from the mean inter-burst interval. A piecewise-linear phase is reconstructed from reference burst onsets:

```math
\phi_i(t)=2\pi\left(n+\frac{t-t_n}{t_{n+1}-t_n}\right), \qquad t\in[t_n,t_{n+1}].
```

The 1:1 phase-locking value is then computed as:

```math
R_{1:1}=\left|\left\langle e^{i(\phi_1(t)-\phi_2(t))}\right\rangle_t\right|.
```

The unwrapped phase difference is also fit with a line to estimate residual phase drift. This avoids falsely labeling two similar but slowly drifting frequencies as synchronized.

A point is classified as `sync_1_1` only if all of the following conditions hold:

- both segments are valid oscillators;
- relative frequency difference is below `FREQ_TOL = 0.025`;
- `R_1_1 >= R_TOL = 0.90`;
- absolute residual phase-drift slope is below `SLOPE_TOL = 0.20 rad/s`.

If the point is not 1:1 synchronized, the code searches for low-order n:m locking relations up to order 4 using:

```math
R_{n:m}=\left|\left\langle e^{i(n\phi_1(t)-m\phi_2(t))}\right\rangle_t\right|.
```

A point is classified as `lock_n_m` only if the strongest non-1:1 relation has high locking value and low residual generalized phase drift. Otherwise, the point is classified as `drift`.

### Resulting frequency

For 1:1 synchronized points, the locked frequency is stored as:

```math
f_{lock}=\frac{f_1+f_2}{2}.
```

This quantity is plotted only inside the 1:1 locking region. Outside this region, the locked-frequency heatmap is masked because a single shared frequency is not meaningful.

### Visualization

`plot_results.py` produces three publication-oriented figures:

1. `network_architecture_equations.png/.pdf` — schematic of the network architecture, model equations, and swept parameters.
2. `arnold_main.png/.pdf` — four-panel summary containing isolated-frequency calibration, regime map, locked-frequency heatmap, and 1:1 phase-locking-value heatmap.
3. `example_traces.png/.pdf` — representative traces for selected regimes. By default, only the final 2 s of the simulation are displayed to make the spikes visible; the run itself uses the full duration stored in the sweep data.

The regime map uses a symlog y-axis for coupling, preserving the exact zero-coupling row while giving high visual resolution to very weak nonzero coupling values.

## Outputs

A complete run produces:

- `sweep_data.npz` — complete machine-readable arrays;
- `sweep_points.csv` — long-form table, one row per parameter point;
- `summary.json` — sweep settings and regime counts;
- `arnold_main.png/.pdf` — main regime-map figure;
- `example_traces.png/.pdf` — short-window example traces;
- `network_architecture_equations.png/.pdf` — architecture/equation schematic;
- `interpretation_summary.json` — regime percentages and interpretation notes;
- `sync_boundary.csv` — empirical minimal coupling needed for 1:1 locking at each detuning;
- `sync_width_by_coupling.csv` — width of the 1:1 locking interval by coupling strength.

## Interpretation

Do not describe the output as a clean analytical Arnold tongue unless the regime map actually has that shape. The correct formulation is:

> We empirically mapped synchronization regimes in two coupled spiking half-center oscillators as a function of isolated frequency mismatch and crossed inter-segment inhibitory strength. The 1:1 phase-locked region is Arnold-tongue-like, but the full spiking model also permits drift, higher-order locking, irregular/asymmetric activity, and loss of valid oscillation.

That wording avoids overclaiming phase-oscillator theory while preserving the useful Arnold-tongue comparison.
