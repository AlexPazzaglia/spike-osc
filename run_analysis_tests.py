"""Minimal validation runner for the CPG synchronization package."""

from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


def run(cmd: list[str], log, timeout: int = 120) -> None:
    log.write("\n$ " + " ".join(cmd) + "\n")
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        env=env,
    )
    log.write(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def main() -> None:
    out = Path("results_tests")
    out.mkdir(exist_ok=True)
    for p in out.glob("*"):
        if p.is_file():
            p.unlink()

    log_path = out / "run_analysis_tests.log"
    with log_path.open("w", encoding="utf-8") as log:

        # Run the sweep
        run(
            [
                sys.executable,
                "sweep_arnold.py",
                "--T", "20.0",
                "--out", str(out),
                "--target-f1-Hz", "3.0",
                "--delta-f-max-Hz", "2.0",
                "--n-delta", "20",
                "--n-g", "20",
            ],
            log,
            timeout=180
        )

        # Run the plotting script
        run(
            [
                sys.executable,
                "plot_results.py",
                "--data",
                str(out / "sweep_data.npz")
            ],
            log,
            timeout=120
        )

        # Run the summary script
        run(
            [
                sys.executable,
                "summarize_results.py",
                "--data",
                str(out / "sweep_data.npz")
            ],
            log,
            timeout=60
        )

        # Check results
        expected = [
            "sweep_data.npz",
            "sweep_points.csv",
            "summary.json",
            "interpretation_summary.json",
            "sync_boundary.csv",
            "sync_width_by_coupling.csv",
            "arnold_main.png",
            "example_traces.png",
            "network_architecture_equations.png",
        ]
        missing = [name for name in expected if not (out / name).exists()]
        if missing:
            raise AssertionError("Missing test outputs: " + ", ".join(missing))

        log.write("\nTest passed.\n")

    print(f"Tests passed. Log: {log_path}")


if __name__ == "__main__":
    main()
