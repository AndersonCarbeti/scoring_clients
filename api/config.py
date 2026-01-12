from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

def _getenv(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name, default)
    return val if val is not None and str(val).strip() != "" else None

@dataclass(frozen=True)
class Settings:
    threshold: float
    fn_cost: float
    fp_cost: float
    model_uri: Optional[str]
    local_model_path: Optional[str]
    clients_csv_path: Optional[str]
    client_id_col: str

def get_settings() -> Settings:
    threshold = float(_getenv("THRESHOLD", "0.5"))
    fn_cost = float(_getenv("FN_COST", "10"))
    fp_cost = float(_getenv("FP_COST", "1"))

    model_uri = _getenv("MODEL_URI")
    local_model_path = _getenv("LOCAL_MODEL_PATH")

    clients_csv_path = _getenv("CLIENTS_CSV_PATH")
    client_id_col = _getenv("CLIENT_ID_COL", "SK_ID_CURR") or "SK_ID_CURR"

    return Settings(
        threshold=threshold,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
        model_uri=model_uri,
        local_model_path=local_model_path,
        clients_csv_path=clients_csv_path,
        client_id_col=client_id_col,
    )
