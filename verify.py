"""Rerun the pipeline in a clean copy of the repository and check every output.

Figures are checked against results/figure_checksums.csv; tables against the committed results/*.csv.
python verify.py --write-checksums records the checksums of the figures currently in results/.
"""
import csv
import glob
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from matplotlib.image import imread

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
CHECKSUMS = os.path.join(RESULTS, "figure_checksums.csv")
SOURCES = ["src", "figures", "data", "reproduce.py"]


def png_digest(path):
    pixels = np.round(np.asarray(imread(path)) * 255).astype(np.uint8)
    return hashlib.sha256(repr(pixels.shape).encode() + pixels.tobytes()).hexdigest()


def write_checksums():
    rows = [(os.path.basename(p), png_digest(p)) for p in sorted(glob.glob(os.path.join(RESULTS, "*.png")))]
    with open(CHECKSUMS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["figure", "sha256"])
        w.writerows(rows)
    print("wrote %d checksums to %s" % (len(rows), CHECKSUMS))


def same_text(a, b):
    with open(a, encoding="utf-8") as fa, open(b, encoding="utf-8") as fb:
        return fa.read() == fb.read()


def main(argv):
    if "--write-checksums" in argv:
        write_checksums()
        return 0
    with open(CHECKSUMS, encoding="utf-8") as f:
        expected = {r["figure"]: r["sha256"] for r in csv.DictReader(f)}
    tables = sorted(os.path.basename(p) for p in glob.glob(os.path.join(RESULTS, "table*.csv")))
    work = tempfile.mkdtemp(prefix="verify_")
    try:
        for name in SOURCES:
            src, dst = os.path.join(ROOT, name), os.path.join(work, name)
            if os.path.isdir(src):
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
            else:
                shutil.copy2(src, dst)
        tmp = os.path.join(work, "tmp")
        os.makedirs(tmp)
        if subprocess.run([sys.executable, "reproduce.py"], cwd=work, env=dict(os.environ, TEMP=tmp, TMP=tmp)).returncode:
            print("reproduce.py failed")
            return 1
        new = os.path.join(work, "results")
        failures = 0
        print()
        for name in sorted(expected):
            p = os.path.join(new, name)
            ok = os.path.exists(p) and png_digest(p) == expected[name]
            failures += not ok
            print("%-44s %s" % (name, "identical" if ok else "differs" if os.path.exists(p) else "missing"))
        for name in tables:
            p = os.path.join(new, name)
            ok = os.path.exists(p) and same_text(p, os.path.join(RESULTS, name))
            failures += not ok
            print("%-44s %s" % (name, "identical" if ok else "differs" if os.path.exists(p) else "missing"))
        print("\n%s" % ("all outputs reproduced" if not failures else "%d outputs differ" % failures))
        return 1 if failures else 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
