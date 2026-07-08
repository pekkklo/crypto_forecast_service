import os
import json
import pickle

from tensorflow.keras.models import load_model

from backend.database import (
    SessionLocal,
    Model,
    Cryptocurrency
)


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


class ModelLoader:

    def __init__(self):
        self.cache = {}

    def resolve_model_path(self, path):
        if os.path.isabs(path):
            return path

        path = path.replace("\\", os.sep)

        if path.startswith("../"):
            path = path[3:]

        return os.path.join(BASE_DIR, path)

    def get_active_model(self, symbol, horizon):
        db = SessionLocal()

        db_model = (
            db.query(Model)
            .join(
                Cryptocurrency,
                Model.coin_id == Cryptocurrency.coin_id
            )
            .filter(
                Cryptocurrency.symbol == symbol,
                Model.horizon == horizon,
                Model.is_active == 1
            )
            .order_by(Model.created_at.desc())
            .first()
        )

        if not db_model:
            db.close()
            return None

        cache_key = f"{symbol}_{horizon}_{db_model.model_id}"

        if cache_key in self.cache:
            db.close()
            return self.cache[cache_key]

        model_path = self.resolve_model_path(
            db_model.model_path
        )

        print("MODEL PATH:", model_path)

        if not os.path.exists(model_path):
            print("MODEL PATH NOT FOUND:", model_path)
            db.close()
            return None

        try:
            metadata_path = os.path.join(
                model_path,
                "metadata.json"
            )

            if not os.path.exists(metadata_path):
                print("METADATA NOT FOUND")
                db.close()
                return None

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Определяем тип модели из метаданных или из db_model
            model_type = metadata.get("model_type") or db_model.model_type
            
            model_data = {
                "model_id": db_model.model_id,
                "symbol": symbol,
                "model_name": db_model.model_name,
                "model_type": model_type,
                "horizon": db_model.horizon,
                "model_path": model_path,
            }

            model_data["feature_cols"] = metadata.get(
                "feature_cols",
                ["volume", "transactions", "active_addresses"]
            )

            model_data["lags"] = metadata.get("lags")
            model_data["lookback"] = metadata.get("lookback", 20)
            
            # =====================================================
            # ВЕСА АНСАМБЛЯ - БЕРУТСЯ ИЗ METADATA
            # =====================================================
            # Проверяем тип ансамбля и загружаем соответствующие веса
            if "xgb_lstm" in model_type or "xgb-lstm" in model_type:
                model_data["xgb_weight"] = metadata.get("xgb_weight", 0.5)
                model_data["lstm_weight"] = metadata.get("lstm_weight", 0.5)
                model_data["tcn_weight"] = None
                print(f"  📊 Ensemble weights: XGB={model_data['xgb_weight']}, LSTM={model_data['lstm_weight']}")
                
            elif "xgb_tcn" in model_type or "xgb-tcn" in model_type:
                model_data["xgb_weight"] = metadata.get("xgb_weight", 0.5)
                model_data["tcn_weight"] = metadata.get("tcn_weight", 0.5)
                model_data["lstm_weight"] = None
                print(f"  📊 Ensemble weights: XGB={model_data['xgb_weight']}, TCN={model_data['tcn_weight']}")
                
            elif "tcn_lstm" in model_type or "tcn-lstm" in model_type:
                model_data["tcn_weight"] = metadata.get("tcn_weight", 0.5)
                model_data["lstm_weight"] = metadata.get("lstm_weight", 0.5)
                model_data["xgb_weight"] = None
                print(f"  📊 Ensemble weights: TCN={model_data['tcn_weight']}, LSTM={model_data['lstm_weight']}")
            else:
                # Fallback: равные веса для всех доступных моделей
                model_data["xgb_weight"] = 0.5
                model_data["lstm_weight"] = 0.5
                model_data["tcn_weight"] = 0.5
                print(f"  📊 Using default weights (0.5 each)")

            # SCALER
            scaler_path = metadata.get(
                "scaler_path",
                os.path.join(model_path, "scaler.pkl")
            )

            scaler_path = self.resolve_model_path(
                scaler_path
            )

            if os.path.exists(scaler_path):
                with open(scaler_path, "rb") as f:
                    model_data["scaler"] = pickle.load(f)
                print("  ✓ Loaded Scaler")
            else:
                model_data["scaler"] = None
                print("  ⚠️ No scaler found")

            # XGB
            xgb_path = os.path.join(model_path, "xgb_model.pkl")
            if os.path.exists(xgb_path):
                with open(xgb_path, "rb") as f:
                    model_data["xgb"] = pickle.load(f)
                print("  ✓ Loaded XGBoost")

            # LSTM
            lstm_path = os.path.join(model_path, "lstm_model.keras")
            if os.path.exists(lstm_path):
                model_data["lstm"] = load_model(lstm_path)
                print("  ✓ Loaded LSTM")

            # TCN
            tcn_path = os.path.join(model_path, "tcn_model.keras")
            if os.path.exists(tcn_path):
                model_data["tcn"] = load_model(tcn_path)
                print("  ✓ Loaded TCN")

            # Проверяем, что загружены все необходимые компоненты
            required_components = []
            if "xgb" in model_type:
                required_components.append("xgb")
            if "lstm" in model_type:
                required_components.append("lstm")
            if "tcn" in model_type:
                required_components.append("tcn")
            
            missing = [c for c in required_components if c not in model_data]
            if missing:
                print(f"  ❌ Missing components: {missing}")
                db.close()
                return None

        except Exception as e:
            print("MODEL LOAD ERROR:", e)
            import traceback
            traceback.print_exc()
            db.close()
            return None

        db.close()
        self.cache[cache_key] = model_data
        return model_data


model_loader = ModelLoader()