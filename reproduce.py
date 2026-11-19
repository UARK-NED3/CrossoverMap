"""Regenerate every thesis figure and table into results/."""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ["src/refit_cal.py", "--check"],
    ["src/make_tables.py"],
    ["figures/device_map.py"],
    ["figures/transition_band.py"],
    ["figures/baseline_crossover.py"],
    ["figures/geometric_sensitivity.py"],
    ["figures/regime_map.py"],
    ["figures/classical_onset.py"],
    ["figures/published_devices.py"],
    ["figures/fittings.py"],
]


def main():
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    start = time.time()
    for step in STEPS:
        t = time.time()
        r = subprocess.run([sys.executable] + step, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
        print("%-36s %-6s %5.0f s" % (" ".join(step), "ok" if r.returncode == 0 else "FAILED", time.time() - t), flush=True)
        if r.returncode:
            print(r.stdout + r.stderr)
            return r.returncode
    print("done in %.0f s; outputs in %s" % (time.time() - start, os.path.join(ROOT, "results")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
