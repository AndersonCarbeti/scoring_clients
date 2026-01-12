from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

import pandas as pd
import mlflow
from mlflow.pyfunc import PyFuncModel

from .config import get_settings

@dataclass(frozen=True)
class LoadedModel:
    model: PyFuncModel
    source: str
    input_columns: Optional[List[str]]

def _infer_input_columns(pyfunc_model: PyFuncModel) -> Optional[List[str]]:
    try:
        schema = pyfunc_model.metadata.get_input_schema()
        if schema is None:
            return None
        return [c.name for c in schema.inputs]  # type: ignore[attr-defined]
    except Exception:
        return None

@lru_cache(maxsize=1)
def load_model() -> LoadedModel:
    s = get_settings()

    if s.model_uri:
        model = mlflow.pyfunc.load_model(s.model_uri)
        cols = _infer_input_columns(model)
        return LoadedModel(model=model, source=f"mlflow_uri:{s.model_uri}", input_columns=cols)

    if s.local_model_path:
        model = mlflow.pyfunc.load_model(s.local_model_path)
        cols = _infer_input_columns(model)
        return LoadedModel(model=model, source=f"local_path:{s.local_model_path}", input_columns=cols)

    raise RuntimeError("No model configured. Set MODEL_URI or LOCAL_MODEL_PATH.")

def features_to_dataframe(features: dict, input_columns: Optional[List[str]]) -> pd.DataFrame:
    df = pd.DataFrame([features])

    if input_columns:
        missing = [c for c in input_columns if c not in df.columns]
        extra = [c for c in df.columns if c not in input_columns]
        if missing:
            raise ValueError(f"Missing required features: {missing[:20]}{'...' if len(missing) > 20 else ''}")
        if extra:
            raise ValueError(f"Unexpected extra features: {extra[:20]}{'...' if len(extra) > 20 else ''}")
        df = df[input_columns]

    return df
