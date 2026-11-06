"""
Fast v2 run on Kaggle: keep training the v1 model on ~100k NEW patents.

Expects the v1 notebook's output attached as input (it holds cpc_model/ and the
v1 data parquet files). Usage inside the notebook:

    !python -u -m cpc_classifier.run_kaggle_v2
"""

import glob
import subprocess
import sys
from pathlib import Path


def find_dir(pattern: str) -> str:
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    if not hits:
        sys.exit(f"could not find {pattern} under /kaggle/input - attach the v1 output")
    return str(Path(hits[0]).parent)


def run(cmd: list[str]):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    v1_model = find_dir("cpc_model/model.pt")
    v1_data = find_dir("cpc_classifier/data/val.parquet")
    print(f"v1 model: {v1_model}\nv1 data:  {v1_data}", flush=True)

    py = [sys.executable, "-u", "-m"]
    run(py + ["cpc_classifier.data_prep", "--extra", f"{v1_data}/train.parquet",
              "--keep-prob", "0.5"])
    run(py + ["cpc_classifier.train",
              "--train-files", f"{v1_data}/train.parquet",
              "cpc_classifier/data/train_extra.parquet",
              "--cpc-only", "--data-dir", v1_data, "--init-from", v1_model,
              "--loss", "asl", "--epochs", "2", "--patience", "1",
              "--batch-size", "64", "--grad-accum", "1",
              "--lr", "1e-5", "--head-lr", "1e-4", "--warmup", "0.03",
              "--out", "/kaggle/working/cpc_model_v2"])


if __name__ == "__main__":
    main()
