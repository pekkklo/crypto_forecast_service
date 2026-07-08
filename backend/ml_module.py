import pandas as pd
import numpy as np

from datetime import date, timedelta
from decimal import Decimal
import logging

from backend.data_catch import DataFetcher
from backend.model_loader import model_loader
from backend.database import Cryptocurrency, Forecast, MarketData

from arch import arch_model

logger = logging.getLogger(__name__)


class MLModule:

    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.garch_models = {}

    # =====================================================
    # GARCH для доверительных интервалов
    # =====================================================

    def get_garch_model(self, symbol, returns):
        """
        Получает или обучает GARCH модель для символа
        """
        cache_key = f"{symbol}_{len(returns)}"
        
        if cache_key in self.garch_models:
            return self.garch_models[cache_key]
        
        # Параметры для разных монет (из ваших расчетов)
        garch_params = {
            'BTC': {'p': 1, 'q': 2, 'model': 'GARCH'},
            'ETH': {'p': 1, 'q': 2, 'model': 'GARCH'},
            'XRP': {'p': 2, 'q': 3, 'o': 1, 'power': 2, 'model': 'GJR-GARCH'},
            'LTC': {'p': 1, 'q': 1, 'o': 1, 'power': 2, 'model': 'GJR-GARCH'},
        }
        
        params = garch_params.get(symbol, {'p': 1, 'q': 1, 'model': 'GARCH'})
        
        try:
            # Очищаем returns от NaN и inf
            clean_returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
            if len(clean_returns) < 50:
                raise ValueError("Not enough clean returns data")
            
            if params['model'] == 'GARCH':
                model = arch_model(
                    clean_returns * 100,
                    vol='Garch',
                    p=params['p'],
                    q=params['q'],
                    dist='normal'
                )
            else:  # GJR-GARCH
                model = arch_model(
                    clean_returns * 100,
                    vol='Garch',
                    p=params['p'],
                    q=params['q'],
                    o=params.get('o', 1),
                    power=params.get('power', 2),
                    dist='normal'
                )
            
            # Обучаем модель
            res = model.fit(update_freq=5, disp='off')
            
            # Прогноз волатильности на 180 дней
            forecast = res.forecast(horizon=180)
            conditional_volatility = np.sqrt(forecast.variance.values[-1]) / 100
            
            # Очищаем от NaN и inf
            conditional_volatility = np.nan_to_num(conditional_volatility, nan=0.02, posinf=0.02)
            conditional_volatility = np.clip(conditional_volatility, 0.005, 0.1)  # 0.5% - 10%
            
            self.garch_models[cache_key] = {
                'model': res,
                'volatility': conditional_volatility
            }
            
            logger.info(f"GARCH model trained for {symbol} (p={params['p']}, q={params['q']})")
            
        except Exception as e:
            logger.warning(f"GARCH failed for {symbol}: {e}, using fallback")
            # Fallback: простая волатильность
            vol = returns.std()
            if np.isnan(vol) or vol < 0.005:
                vol = 0.02  # 2% по умолчанию
            self.garch_models[cache_key] = {
                'model': None,
                'volatility': np.full(180, vol)
            }
        
        return self.garch_models[cache_key]

    def compute_confidence_intervals(self, symbol, predictions, horizon, returns=None):
        """
        Вычисляет доверительные интервалы на основе GARCH
        """
        if returns is None or len(returns) < 100:
            # Fallback: простая волатильность с нарастающей неопределенностью
            daily_vol = 0.02  # 2% по умолчанию
            cumulative_vol = daily_vol * np.sqrt(np.arange(1, horizon + 1))
        else:
            # Получаем GARCH прогноз волатильности
            garch_result = self.get_garch_model(symbol, returns)
            garch_vol = garch_result['volatility']
            
            # Если GARCH вернул меньше значений, дополняем
            if len(garch_vol) < horizon:
                last_vol = garch_vol[-1] if len(garch_vol) > 0 else 0.02
                garch_vol = np.append(garch_vol, [last_vol] * (horizon - len(garch_vol)))
            else:
                garch_vol = garch_vol[:horizon]
            
            # Кумулятивная волатильность (накопленная неопределенность)
            cumulative_vol = np.sqrt(np.cumsum(np.maximum(garch_vol, 0.005) ** 2))
        
        # 95% доверительный интервал: +-1.96 * sigma
        # Для цен используем мультипликативный подход: price * exp(+-1.96 * sigma * t^0.5)
        uncertainty = 1.96 * cumulative_vol
        
        # Доверительные интервалы
        lower_bounds = predictions * np.exp(-uncertainty)
        upper_bounds = predictions * np.exp(uncertainty)
        
        return lower_bounds, upper_bounds

    # =====================================================
    # построение последовательностей для LSTM и TCN
    # =====================================================

    def build_sequence(self, df, lookback, feature_cols, scaler=None):
        """
        Строит последовательность для LSTM/TCN с масштабированием
        """
        # Проверяем, что данных достаточно
        if len(df) < lookback:
            raise ValueError(f"Not enough data: need {lookback}, have {len(df)}")
        
        # Берем последние lookback строк
        cols = ['close'] + [c for c in feature_cols if c in df.columns]
        last_data = df[cols].values[-lookback:]
        
        if scaler is not None:
            last_data = scaler.transform(last_data)
        
        return last_data.reshape(1, lookback, -1)

    def build_xgb_features(self, df, lags, feature_cols):
        """
        Строит фичи для XGBoost (лаги цены и фич)
        """
        price = df['close'].values
        
        if len(price) < lags:
            return None
        
        row = []
        row.extend(price[-lags:])
        
        for col in feature_cols:
            if col in df.columns:
                row.extend(df[col].values[-lags:])
            else:
                row.extend([0] * lags)
        
        return np.array([row])

    # =====================================================
    # XGB прогнозы
    # =====================================================

    def predict_xgb_multi(self, model_data, df):
        lags = model_data.get("lags")
        if not lags:
            logger.warning("No lags parameter in model_data")
            return None
            
        feature_cols = model_data["feature_cols"]
        
        X = self.build_xgb_features(df, lags, feature_cols)
        
        if X is None:
            return None
        
        predictions = model_data["xgb"].predict(X)[0]
        return predictions

    # =====================================================
    # LSTM прогнозы
    # =====================================================

    def predict_lstm_multi(self, model_data, df):
        lookback = model_data["lookback"]
        feature_cols = model_data["feature_cols"]
        scaler = model_data.get("scaler")
        
        seq = self.build_sequence(df, lookback, feature_cols, scaler)
        predictions_scaled = model_data["lstm"].predict(seq, verbose=0)[0]
        
        if scaler is not None:
            n_features = len(feature_cols) + 1
            dummy = np.zeros((len(predictions_scaled), n_features))
            dummy[:, 0] = predictions_scaled
            predictions = scaler.inverse_transform(dummy)[:, 0]
        else:
            predictions = predictions_scaled
        
        return predictions

    # =====================================================
    # TCN прогнозы
    # =====================================================

    def predict_tcn_multi(self, model_data, df):
        lookback = model_data["lookback"]
        feature_cols = model_data["feature_cols"]
        scaler = model_data.get("scaler")
        
        seq = self.build_sequence(df, lookback, feature_cols, scaler)
        predictions_scaled = model_data["tcn"].predict(seq, verbose=0)[0]
        
        if scaler is not None:
            n_features = len(feature_cols) + 1
            dummy = np.zeros((len(predictions_scaled), n_features))
            dummy[:, 0] = predictions_scaled
            predictions = scaler.inverse_transform(dummy)[:, 0]
        else:
            predictions = predictions_scaled
        
        return predictions

    # =====================================================
    # Гибридные прогнозы
    # =====================================================

    def predict_with_xgb_lstm(self, model_data, df):
        xgb_pred = self.predict_xgb_multi(model_data, df)
        lstm_pred = self.predict_lstm_multi(model_data, df)
        
        if xgb_pred is None:
            return lstm_pred
        if lstm_pred is None:
            return xgb_pred
        
        # Проверяем, что прогнозы одной длины
        min_len = min(len(xgb_pred), len(lstm_pred))
        xgb_pred = xgb_pred[:min_len]
        lstm_pred = lstm_pred[:min_len]
        
        xgb_weight = model_data.get("xgb_weight", 0.5)
        lstm_weight = model_data.get("lstm_weight", 0.5)
        
        # Нормализация: сохраняем пропорции, но сумма должна быть 1
        total_weight = xgb_weight + lstm_weight
        if total_weight > 0:
            ensemble = (xgb_weight * xgb_pred + lstm_weight * lstm_pred) / total_weight
        else:
            ensemble = (xgb_pred + lstm_pred) / 2
        
        return np.maximum(ensemble, 0.01)

    def predict_with_xgb_tcn(self, model_data, df):
        xgb_pred = self.predict_xgb_multi(model_data, df)
        tcn_pred = self.predict_tcn_multi(model_data, df)
        
        if xgb_pred is None:
            return tcn_pred
        if tcn_pred is None:
            return xgb_pred
        
        min_len = min(len(xgb_pred), len(tcn_pred))
        xgb_pred = xgb_pred[:min_len]
        tcn_pred = tcn_pred[:min_len]
        
        xgb_weight = model_data.get("xgb_weight", 0.5)
        tcn_weight = model_data.get("tcn_weight", 0.5)
        
        total_weight = xgb_weight + tcn_weight
        if total_weight > 0:
            ensemble = (xgb_weight * xgb_pred + tcn_weight * tcn_pred) / total_weight
        else:
            ensemble = (xgb_pred + tcn_pred) / 2
        
        return np.maximum(ensemble, 0.01)

    def predict_with_tcn_lstm(self, model_data, df):
        tcn_pred = self.predict_tcn_multi(model_data, df)
        lstm_pred = self.predict_lstm_multi(model_data, df)
        
        if tcn_pred is None:
            return lstm_pred
        if lstm_pred is None:
            return tcn_pred
        
        min_len = min(len(tcn_pred), len(lstm_pred))
        tcn_pred = tcn_pred[:min_len]
        lstm_pred = lstm_pred[:min_len]
        
        tcn_weight = model_data.get("tcn_weight", 0.5)
        lstm_weight = model_data.get("lstm_weight", 0.5)
        
        total_weight = tcn_weight + lstm_weight
        if total_weight > 0:
            ensemble = (tcn_weight * tcn_pred + lstm_weight * lstm_pred) / total_weight
        else:
            ensemble = (tcn_pred + lstm_pred) / 2
        
        return np.maximum(ensemble, 0.01)

    # =====================================================
    # Обнолвение прогнозов
    # =====================================================

    def update_all(self, db):
        cryptos = db.query(Cryptocurrency).all()
        today = date.today()
        total_forecasts = 0

        for crypto in cryptos:
            symbol = f"{crypto.symbol}/USDT"
            logger.info(f"Updating {symbol}")

            self.data_fetcher.update_market_data(
                db,
                symbol,
                crypto.coin_id,
                365*3
            )

            market_data = (
                db.query(MarketData)
                .filter(MarketData.coin_id == crypto.coin_id)
                .order_by(MarketData.date)
                .all()
            )

            if len(market_data) < 100:
                logger.warning(f"Not enough data for {symbol} (need 100+, have {len(market_data)})")
                continue

            df = pd.DataFrame([{
                "close": float(m.close_price),
                "volume": float(m.volume or 0),
                "active_addresses": m.active_addresses or 0,
                "tx_count": m.transactions or 0                
            } for m in market_data])

            # Доходности для GARCH
            returns = np.diff(np.log(df['close'].values + 1e-8))
            returns = np.nan_to_num(returns, nan=0.0, posinf=0.02, neginf=-0.02)

            for horizon in [30, 180]:
                model_data = model_loader.get_active_model(
                    crypto.symbol,
                    horizon
                )

                if not model_data:
                    logger.warning(f"No model {crypto.symbol} {horizon}")
                    continue

                logger.info(
                    f"Loaded model {crypto.symbol} "
                    f"{model_data['model_type']} "
                    f"{horizon}"
                )

                try:
                    model_type = model_data["model_type"]

                    if model_type in ["xgb_lstm", "xgb-lstm"]:
                        predictions = self.predict_with_xgb_lstm(model_data, df)
                    elif model_type in ["xgb_tcn", "xgb-tcn"]:
                        predictions = self.predict_with_xgb_tcn(model_data, df)
                    elif model_type in ["tcn_lstm", "tcn-lstm"]:
                        predictions = self.predict_with_tcn_lstm(model_data, df)
                    else:
                        logger.warning(f"Unknown model type: {model_type}")
                        continue

                    if predictions is None:
                        logger.warning(f"No predictions for {crypto.symbol} {horizon}")
                        continue
                    
                    if len(predictions) < horizon:
                        logger.warning(f"Predictions length {len(predictions)} < horizon {horizon}")
                        continue

                    predictions = predictions[:horizon]
                    
                    # Доверительные интервалы через GARCH
                    lower_bounds, upper_bounds = self.compute_confidence_intervals(
                        symbol=crypto.symbol,
                        predictions=predictions,
                        horizon=horizon,
                        returns=returns
                    )

                    # Просто проверяем, есть ли уже прогнозы за сегодня
                    existing_today = db.query(Forecast).filter(
                        Forecast.coin_id == crypto.coin_id,
                        Forecast.model_id == model_data["model_id"],
                        Forecast.forecast_date == today
                    ).first()

                    if existing_today:
                        logger.info(f"Forecasts for {crypto.symbol} {horizon} already exist for today, skipping")
                        continue

                    # Создаем НОВЫЕ прогнозы (старые остаются в БД для метрик)
                    for i, (pred_price, lower, upper) in enumerate(zip(predictions, lower_bounds, upper_bounds), start=1):
                        pred_price = max(float(pred_price), 0.01)
                        lower = max(float(lower), 0.01, pred_price * 0.5)
                        upper = max(float(upper), pred_price * 1.01)

                        db.add(
                            Forecast(
                                coin_id=crypto.coin_id,
                                model_id=model_data["model_id"],
                                forecast_date=today,
                                target_date=today + timedelta(days=i),
                                predicted_price=Decimal(str(pred_price)),
                                lower_bound=Decimal(str(lower)),
                                upper_bound=Decimal(str(upper))
                            )
                        )

                    db.commit()
                    
                    avg_width = ((upper_bounds / lower_bounds).mean() - 1) * 100
                    logger.info(
                        f"Added {horizon} NEW forecasts for {crypto.symbol} "
                        f"(kept old ones for metrics, GARCH interval width: {avg_width:.1f}%)"
                    )
                    total_forecasts += horizon

                except Exception as e:
                    logger.exception(f"Error processing {crypto.symbol} {horizon}: {e}")
                    db.rollback()
                    continue

        logger.info(f"TOTAL forecasts added today: {total_forecasts}")
        return {"status": "ok", "total_forecasts": total_forecasts}