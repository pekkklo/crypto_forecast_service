from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import date
from pydantic import BaseModel
import logging

from backend.database import get_db, Base, engine
from backend.database import (
    Cryptocurrency,
    Forecast,
    Model,
    ModelMetric,
    MarketData,
    Admin
)
from backend.ml_module import MLModule

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Crypto Forecast API")

ml_module = MLModule()

logger = logging.getLogger(__name__)


class UpdateRequest(BaseModel):
    login: str
    password: str


# =========================
# Список монет
# =========================
@app.get("/api/cryptocurrencies")
def get_cryptocurrencies(db: Session = Depends(get_db)):
    return db.query(Cryptocurrency).all()


# =========================
# Актуальные прогнозы
# =========================
@app.get("/api/forecasts/{symbol}")
def get_forecasts(symbol: str, horizon: int = 30, db: Session = Depends(get_db)):

    crypto = db.query(Cryptocurrency).filter(
        Cryptocurrency.symbol == symbol.upper()
    ).first()

    if not crypto:
        raise HTTPException(404, "Cryptocurrency not found")

    model = db.query(Model).filter(
        Model.coin_id == crypto.coin_id,
        Model.horizon == horizon,
        Model.is_active == 1
    ).order_by(Model.created_at.desc()).first()

    if not model:
        raise HTTPException(404, "Model not found")

    today = date.today()

    forecasts = db.query(Forecast).filter(
        Forecast.coin_id == crypto.coin_id,
        Forecast.model_id == model.model_id,
        Forecast.target_date > today
    ).order_by(Forecast.target_date).all()

    return {
        "symbol": crypto.symbol,
        "forecasts": [
            {
                "target_date": f.target_date,
                "predicted_price": float(f.predicted_price),
                "lower_bound": float(f.lower_bound),
                "upper_bound": float(f.upper_bound)
            }
            for f in forecasts
        ]
    }


# =========================
# Метрики
# =========================
@app.get("/api/metrics/{symbol}")
def get_metrics(symbol: str, db: Session = Depends(get_db)):

    crypto = db.query(Cryptocurrency).filter(
        Cryptocurrency.symbol == symbol.upper()
    ).first()

    if not crypto:
        raise HTTPException(404, "Cryptocurrency not found")

    results = []

    for model in crypto.models:
        for metric in model.metrics:
            results.append({
                "model": model.model_name,
                "model_type": model.model_type,
                "horizon": model.horizon,
                "calculated_at": metric.calculated_at,
                "mae": float(metric.mae) if metric.mae else None,
                "mape": float(metric.mape) if metric.mape else None
            })

    return results


# =========================
# Исторические цены
# =========================
@app.get("/api/historical/{symbol}")
def get_historical(symbol: str, days: int = 365, db: Session = Depends(get_db)):

    crypto = db.query(Cryptocurrency).filter(
        Cryptocurrency.symbol == symbol.upper()
    ).first()

    if not crypto:
        raise HTTPException(404, "Cryptocurrency not found")

    data = db.query(MarketData).filter(
        MarketData.coin_id == crypto.coin_id
    ).order_by(MarketData.date).all()

    return [
        {"date": d.date, "price": float(d.close_price)}
        for d in data
    ]
    
# =========================================================
# Прошлые прогнозы
# =========================================================

@app.get("/api/forecast-history/{symbol}")
def get_forecast_history(
    symbol: str,
    horizon: int = 30,
    db: Session = Depends(get_db)
):

    crypto = db.query(Cryptocurrency).filter(
        Cryptocurrency.symbol == symbol.upper()
    ).first()

    if not crypto:
        raise HTTPException(404, "Cryptocurrency not found")

    models = db.query(Model).filter(
        Model.coin_id == crypto.coin_id,
        Model.horizon == horizon
    ).all()

    if not models:
        raise HTTPException(404, "Models not found")

    model_ids = [m.model_id for m in models]

    forecasts = db.query(Forecast).filter(
        Forecast.coin_id == crypto.coin_id,
        Forecast.model_id.in_(model_ids)
    ).order_by(
        Forecast.target_date,
        Forecast.forecast_date
    ).all()

    result = []

    for f in forecasts:

        days_diff = (
            f.target_date - f.forecast_date
        ).days

        if days_diff != horizon:
            continue

        result.append({
            "forecast_date": f.forecast_date,
            "target_date": f.target_date,
            "predicted_price": float(f.predicted_price),
            "lower_bound": float(f.lower_bound),
            "upper_bound": float(f.upper_bound),
            "model_id": f.model_id
        })

    return result

# =========================
# Запуск обновления данных
# =========================
@app.post("/api/admin/update")
def admin_update(data: UpdateRequest, db: Session = Depends(get_db)):

    admin = db.query(Admin).filter(
        Admin.login == data.login
    ).first()

    if not admin or admin.password != data.password:
        raise HTTPException(401, "Invalid credentials")

    logger.info("Starting ML pipeline")

    try:
        result = ml_module.update_all(db)
        logger.info("ML pipeline finished")
        return result

    except Exception as e:
        logger.exception(e)
        raise HTTPException(500, "Pipeline failed")


if __name__ == "__main__":
    import uvicorn

    print("STARTING UVICORN SERVER...")

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )