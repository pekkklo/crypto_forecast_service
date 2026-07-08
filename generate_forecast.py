"""
Генерация исторических прогнозов "задним числом" для демонстрации
Запуск: python generate_forecast.py
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.database import (
    SessionLocal,
    Cryptocurrency,
    Model,
    Forecast,
    MarketData
)

from datetime import date, timedelta
from decimal import Decimal
import numpy as np

db = SessionLocal()

# =========================================================
# ПЕРИОД: с 2024 года по сегодня
# =========================================================

end_date = date.today()
start_date = date(2024, 1, 1)

print(f"Генерируем прогнозы с {start_date} по {end_date}")

cryptos = db.query(Cryptocurrency).all()

for crypto in cryptos:

    print(f"\n📊 {crypto.symbol}")

    models = db.query(Model).filter(
        Model.coin_id == crypto.coin_id
    ).all()

    if not models:
        print(f"  ⚠️ Нет моделей для {crypto.symbol}")
        continue

    market_data = db.query(MarketData).filter(
        MarketData.coin_id == crypto.coin_id,
        MarketData.date >= start_date - timedelta(days=180)
    ).order_by(MarketData.date).all()

    if len(market_data) < 60:
        print(f"  ⚠️ Недостаточно исторических данных")
        continue

    price_dict = {
        m.date: float(m.close_price)
        for m in market_data
    }

    for model in models:

        horizon = model.horizon

        print(
            f"  🎯 Модель {model.model_name} "
            f"(horizon={horizon})"
        )

        # =====================================================
        # ВАЖНО:
        # Для horizon=180 нельзя генерировать forecast_date
        # ближе чем за 180 дней до today
        # =====================================================

        max_forecast_date = end_date - timedelta(days=horizon)

        current_date = start_date

        forecasts_created = 0

        while current_date <= max_forecast_date:

            # =================================================
            # ПРОВЕРКА СУЩЕСТВУЮЩЕГО ПРОГНОЗА
            # =================================================

            existing = db.query(Forecast).filter(
                Forecast.model_id == model.model_id,
                Forecast.forecast_date == current_date
            ).first()

            if existing:
                current_date += timedelta(days=1)
                continue

            base_price = price_dict.get(current_date)

            if not base_price:
                current_date += timedelta(days=1)
                continue

            # =================================================
            # ГЕНЕРАЦИЯ FORECASTS
            # =================================================

            for i in range(1, horizon + 1):

                target_date = current_date + timedelta(days=i)

                if target_date > end_date:
                    continue

                actual_price = price_dict.get(target_date)

                # =============================================
                # ЕСЛИ ЕСТЬ ИСТОРИЧЕСКАЯ ЦЕНА —
                # делаем realistic prediction
                # =============================================

                if actual_price:

                    error_pct = np.random.normal(
                        0,
                        0.03 * np.sqrt(i / 30)
                    )

                    predicted_price = (
                        actual_price * (1 + error_pct)
                    )

                else:

                    # =========================================
                    # FALLBACK TREND
                    # =========================================

                    days_ahead = (
                        target_date - current_date
                    ).days

                    trend = 0.0005

                    predicted_price = (
                        base_price * (1 + trend) ** days_ahead
                    )

                # =============================================
                # CONFIDENCE INTERVAL
                # =============================================

                volatility = 0.03 * np.sqrt(i)

                lower_bound = (
                    predicted_price *
                    (1 - 1.96 * volatility)
                )

                upper_bound = (
                    predicted_price *
                    (1 + 1.96 * volatility)
                )

                forecast = Forecast(
                    coin_id=crypto.coin_id,
                    model_id=model.model_id,
                    forecast_date=current_date,
                    target_date=target_date,
                    predicted_price=Decimal(
                        str(round(predicted_price, 2))
                    ),
                    lower_bound=Decimal(
                        str(round(lower_bound, 2))
                    ),
                    upper_bound=Decimal(
                        str(round(upper_bound, 2))
                    )
                )

                db.add(forecast)

                forecasts_created += 1

            current_date += timedelta(days=1)

            # =================================================
            # BATCH COMMIT
            # =================================================

            if forecasts_created > 0 and forecasts_created % 500 == 0:

                db.commit()

                print(
                    f"    Создано "
                    f"{forecasts_created} прогнозов..."
                )

        db.commit()

        print(
            f"    ✅ Создано "
            f"{forecasts_created} прогнозов "
            f"для {model.model_name}"
        )

print("\n🎉 Генерация исторических прогнозов завершена!")

db.close()