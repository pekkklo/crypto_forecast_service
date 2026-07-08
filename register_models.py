# register_models.py

import os
import sys
import json
from datetime import date

# ==================================================
# PROJECT ROOT
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# ==================================================
# IMPORTS
# ==================================================

from backend.database import (
    SessionLocal,
    Cryptocurrency,
    Model,
    ModelMetric,
    GarchConfig
)

# ==================================================
# DB
# ==================================================

db = SessionLocal()

# ==================================================
# COINS
# ==================================================

coins = {
    c.symbol.upper(): c
    for c in db.query(Cryptocurrency).all()
}

# ==================================================
# MODELS DIR
# ==================================================

MODELS_DIR = os.path.join(BASE_DIR, "models")

# ==================================================
# AUTO DISCOVER MODELS
# ==================================================

model_folders = [
    folder for folder in os.listdir(MODELS_DIR)
    if os.path.isdir(os.path.join(MODELS_DIR, folder))
]

print(f"Found {len(model_folders)} model folders")

# ==================================================
# REGISTER MODELS
# ==================================================

for folder in model_folders:

    model_dir = os.path.join(MODELS_DIR, folder)
    metadata_path = os.path.join(model_dir, "metadata.json")

    # ----------------------------------------------
    # CHECK METADATA
    # ----------------------------------------------

    if not os.path.exists(metadata_path):
        print(f"metadata.json missing in {folder}")
        continue

    # ----------------------------------------------
    # LOAD METADATA
    # ----------------------------------------------

    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    except Exception as e:
        print(f"Failed reading metadata for {folder}: {e}")
        continue

    # ----------------------------------------------
    # METADATA FIELDS
    # ----------------------------------------------

    symbol = metadata.get("symbol", "").upper()

    model_name = metadata.get("model_name")

    model_type = metadata.get("model_type")

    # NORMALIZE MODEL TYPE
    model_type = model_type.replace("-", "_")

    horizon = metadata.get("horizon", 180)

    lags = metadata.get("lags")

    lookback = metadata.get("lookback")

    # ABSOLUTE PATHS
    model_path = os.path.abspath(model_dir)

    scaler_path = os.path.join(model_path, "scaler.pkl")

    test_mape = metadata.get("test_mape")

    test_mae = metadata.get("test_mae")

    n_samples = metadata.get("n_samples")

    is_active = metadata.get("is_active", 1)

    # ----------------------------------------------
    # VALIDATE
    # ----------------------------------------------

    if symbol not in coins:
        print(f"Unknown symbol: {symbol}")
        continue

    coin = coins[symbol]

    # ==================================================
    # DEACTIVATE OLD MODELS
    # ==================================================

    db.query(Model).filter(
        Model.coin_id == coin.coin_id,
        Model.horizon == horizon
    ).update({"is_active": 0})

    # ==================================================
    # FIND EXISTING
    # ==================================================

    existing = db.query(Model).filter(
        Model.coin_id == coin.coin_id,
        Model.model_name == model_name,
        Model.horizon == horizon
    ).first()

    # ==================================================
    # UPDATE / CREATE
    # ==================================================

    if existing:

        existing.model_type = model_type
        existing.lags = lags
        existing.lookback = lookback
        existing.model_path = model_path
        existing.scaler_path = scaler_path
        existing.is_active = is_active

        model_record = existing

        print(f"Updated model: {model_name}")

    else:

        model_record = Model(
            coin_id=coin.coin_id,
            model_name=model_name,
            model_type=model_type,
            horizon=horizon,
            lags=lags,
            lookback=lookback,
            model_path=model_path,
            scaler_path=scaler_path,
            created_at=date.today(),
            is_active=is_active
        )

        db.add(model_record)

        print(f"Added model: {model_name}")

    db.commit()

    # ==================================================
    # METRICS
    # ==================================================

    existing_metric = db.query(ModelMetric).filter(
        ModelMetric.model_id == model_record.model_id
    ).first()

    if existing_metric:

        existing_metric.mae = test_mae
        existing_metric.mape = test_mape
        existing_metric.n_samples = n_samples
        existing_metric.calculated_at = date.today()

        print(f"Updated metrics for {model_name}")

    else:

        metric = ModelMetric(
            model_id=model_record.model_id,
            calculated_at=date.today(),
            mae=test_mae,
            mape=test_mape,
            n_samples=n_samples
        )

        db.add(metric)

        print(f"Added metrics for {model_name}")

    db.commit()

# ==================================================
# GARCH CONFIGS
# ==================================================

garch_configs = [
    ("BTC", "GARCH", 1, 2, 0, 2),
    ("ETH", "GARCH", 1, 2, 0, 2),
    ("XRP", "GJR-GARCH", 2, 3, 1, 2),
    ("LTC", "GJR-GARCH", 1, 1, 1, 2),
]

for symbol, model_type, p, q, o, power in garch_configs:

    coin = coins[symbol]

    existing = db.query(GarchConfig).filter(
        GarchConfig.coin_id == coin.coin_id
    ).first()

    if existing:

        existing.model_type = model_type
        existing.p = p
        existing.q = q
        existing.o = o
        existing.power = power

        print(f"Updated GARCH for {symbol}")

    else:

        db.add(
            GarchConfig(
                coin_id=coin.coin_id,
                model_type=model_type,
                p=p,
                q=q,
                o=o,
                power=power
            )
        )

        print(f"Added GARCH for {symbol}")

db.commit()
db.close()

print("\nALL MODELS REGISTERED")