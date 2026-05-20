"""Small tests to verify that simulator, sweep, plotting, and summaries run.

All generated files, including the smoke-test log, are written inside the
``smoke_results`` directory. Nothing is written to the package root except the
source code itself.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from analysis import analyze_segment, evaluate_point
from cpg_sim import NUMBA_AVAILABLE, run_sim


class Tee:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.fh = log_path.open("w", encoding="utf-8")

    def write(self, msg: str) -> None:
        print(msg, end="", flush=True)
        self.fh.write(msg)
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def assert_true(cond, msg):
    if not cond:
        raise AssertionError(msg)


def run_logged(cmd: list[str], *, timeout: int, tee: Tee) -> None:
    tee.write("\n$ " + " ".join(cmd) + "\n")
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    tee.write(proc.stdout)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)


def main():
    out = Path("smoke_results")
    if out.exists():
        for p in out.glob("*"):
            if p.is_file():
                p.unlink()
    out.mkdir(exist_ok=True)
    tee = Tee(out / "run_smoke_tests.log")

    try:
        tee.write(f"Numba available: {NUMBA_AVAILABLE}\n")

        params = {"Delta_w": 25e-12, "I_ext": 240e-12}

        # With the current model, k≈0.5 gives a ~3 Hz reference oscillator.
        spk = run_sim(T=8.0, k1=0.5, k2=0.5, g_inter=0.0, params=params)
        a1 = analyze_segment(spk[0], spk[1], t_start=2.0, isi_gap=0.15 * 0.5)
        a2 = analyze_segment(spk[2], spk[3], t_start=2.0, isi_gap=0.15 * 0.5)
        tee.write(f"isolated seg1: {a1['status']} {a1['freq']} Hz\n")
        tee.write(f"isolated seg2: {a2['status']} {a2['freq']} Hz\n")
        assert_true(a1["status"] == "osc", "Segment 1 should oscillate in isolation")
        assert_true(a2["status"] == "osc", "Segment 2 should oscillate in isolation")
        assert_true(2.5 <= float(a1["freq"]) <= 3.5, "Reference frequency should be close to 3 Hz")

        r0 = evaluate_point(k1=0.5, k2=0.5, g_inter=10e-9, T=8.0, params=params)
        tee.write(
            "coupled k1=k2=0.5 g=10nS: "
            f"{r0['regime']} {r0['freq1']} {r0['freq2']} {r0['R_1_1']}\n"
        )
        assert_true(r0["status1"] == "osc" and r0["status2"] == "osc", "Coupled segments should oscillate")

        cmd = [
            sys.executable, "sweep_arnold.py", "--quick", "--T", "8", "--out", str(out),
            "--target-f1-Hz", "3.0", "--delta-f-max-Hz", "1.8", "--n-delta", "19", "--n-g", "13",
        ]
        tee.write("running quick sweep...\n")
        run_logged(cmd, timeout=90, tee=tee)

        tee.write("plotting...\n")
        run_logged([sys.executable, "plot_results.py", "--data", str(out / "sweep_data.npz")], timeout=90, tee=tee)

        tee.write("summarizing...\n")
        run_logged([sys.executable, "summarize_results.py", "--data", str(out / "sweep_data.npz")], timeout=60, tee=tee)

        for fname in [
            "sweep_data.npz", "sweep_points.csv", "summary.json", "interpretation_summary.json",
            "sync_boundary.csv", "sync_width_by_coupling.csv", "arnold_main.png",
            "example_traces.png", "network_architecture_equations.png", "run_smoke_tests.log",
        ]:
            assert_true((out / fname).exists(), f"Missing {fname}")

        with (out / "summary.json").open() as f:
            summary = json.load(f)
        tee.write("summary:\n" + json.dumps(summary, indent=2) + "\n")
        tee.write("Smoke tests passed.\n")
    finally:
        tee.close()


if __name__ == "__main__":
    main()
