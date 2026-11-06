"""
Build the 100k prior-art index on Kaggle (GPU). Usage inside the notebook:

    !python -u -m retrieval.run_kaggle_index
"""

import subprocess
import sys

OUT = "/kaggle/working/prior_art_index"


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    run([sys.executable, "-m", "pip", "install", "-q", "bm25s"])
    py = [sys.executable, "-u", "-m"]
    run(py + ["retrieval.build_corpus", "--per-year", "20000", "--out", OUT])
    run(py + ["retrieval.build_index", "--index", OUT, "--batch-size", "128"])
