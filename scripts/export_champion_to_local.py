"""Export the MLflow registry 'champion' to a local MLflow model folder.

If your cloud runtime cannot reach your MLflow registry, you can export the champion model
to `artifacts/model` and commit it (or attach it as build artifact).

Usage:
  export MLFLOW_TRACKING_URI=...
  python scripts/export_champion_to_local.py --model-uri models:/home_credit_default_model@champion --out artifacts/model
"""

from __future__ import annotations

import argparse
from pathlib import Path
import mlflow

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-uri", required=True, help="e.g., models:/home_credit_default_model@champion")
    ap.add_argument("--out", default="artifacts/model", help="Local output directory")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    model = mlflow.pyfunc.load_model(args.model_uri)
    # Save full MLflow model (pyfunc flavor) locally
    mlflow.pyfunc.save_model(path=str(out), python_model=model._model_impl)  # best effort
    print(f"Saved {args.model_uri} to {out}")

if __name__ == "__main__":
    main()
