import ccxt
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcher:
    """Класс для получения данных с бирж и on-chain метрик"""
    
    def __init__(self):
        self.exchange = ccxt.kucoin()
        self.exchange.load_markets()
        
        # Маппинг символов KuCoin -> CoinMetrics
        self.symbols_map = {
            "BTC/USDT": "btc",
            "ETH/USDT": "eth",
            "XRP/USDT": "xrp",
            "LTC/USDT": "ltc"
        }
        
        self.TIMEFRAME = "1d"
        self.LIMIT = 1000
    
    def fetch_ohlcv_range(self, symbol: str, days: int) -> pd.DataFrame:
        """Получение OHLCV данных с KuCoin за указанный период"""
        since = self.exchange.parse8601(
            (datetime.utcnow() - timedelta(days=days))
            .strftime("%Y-%m-%dT%H:%M:%S")
        )
        
        all_candles = []
        
        while True:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=self.TIMEFRAME,
                    since=since,
                    limit=self.LIMIT
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                since = candles[-1][0] + 1
                
                if len(candles) < self.LIMIT:
                    break
                
                time.sleep(self.exchange.rateLimit / 1000)
            except Exception as e:
                logger.error(f"Error fetching OHLCV for {symbol}: {e}")
                break
        
        if all_candles:
            df = pd.DataFrame(
                all_candles,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
            return df[["date", "close", "volume"]]
        
        return pd.DataFrame()
    
    def fetch_onchain_metrics(self, asset: str, days: int) -> pd.DataFrame:
        """Получение on-chain метрик с CoinMetrics"""
        metrics = "AdrActCnt, TxCnt"
        base_url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
        df = pd.DataFrame(columns=["date", "active_addresses", "tx_count"])
        
        params = {
            "assets": asset,
            "metrics": metrics,
            "start_time": (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d"),
            "frequency": "1d",
            "pretty": "true"
        }
        
        current_url = base_url
        
        while True:
            try:
                r = requests.get(current_url, params=params, timeout=30)
                r.raise_for_status()
                response_json = r.json()
                
                if "data" in response_json and response_json["data"]:
                    temp_df = pd.DataFrame(response_json["data"])
                    temp_df["date"] = pd.to_datetime(temp_df["time"]).dt.date
                    temp_df = temp_df.rename(columns={
                        "AdrActCnt": "active_addresses",
                        "TxCnt": "tx_count"
                    })
                    
                    df = pd.concat([df, temp_df[["date", "active_addresses", "tx_count"]]], ignore_index=True)
                
                current_url = response_json.get('next_page_url')
                if not current_url:
                    break
                    
                params = {}
                
            except Exception as e:
                logger.error(f"Error fetching onchain metrics for {asset}: {e}")
                break
        
        if not df.empty:
            df["active_addresses"] = pd.to_numeric(df["active_addresses"], errors='coerce')
            df["tx_count"] = pd.to_numeric(df["tx_count"], errors='coerce')
            df = df.sort_values("date").reset_index(drop=True)
        
        return df
    
    def get_merged_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Получение объединенных данных (OHLCV + on-chain)"""
        if symbol not in self.symbols_map:
            return pd.DataFrame()
        
        asset = self.symbols_map[symbol]
        
        price_df = self.fetch_ohlcv_range(symbol, days)
        if price_df.empty:
            return pd.DataFrame()
        
        onchain_df = self.fetch_onchain_metrics(asset, days)
        
        if not onchain_df.empty:
            merged_df = pd.merge(price_df, onchain_df, on='date', how='left')
        else:
            merged_df = price_df.copy()
            merged_df['active_addresses'] = None
            merged_df['tx_count'] = None
        
        return merged_df
    
    def update_market_data(self, db, symbol, coin_id, days_back=365):
        """Обновление рыночных данных в БД"""
        from backend.database import MarketData        
        merged_df = self.get_merged_data(symbol, days_back)
        
        if merged_df.empty:
            logger.error(f"No data fetched for {symbol}")
            return 0
        
        existing_dates = set()
        existing = db.query(MarketData).filter(
            MarketData.coin_id == coin_id
        ).all()
        
        for record in existing:
            existing_dates.add(record.date)
        
        added_count = 0
        for _, row in merged_df.iterrows():
            if row['date'] not in existing_dates:
                market_data = MarketData(
                    coin_id=coin_id,
                    date=row['date'],
                    close_price=Decimal(str(row['close'])) if pd.notna(row['close']) else None,
                    volume=Decimal(str(row['volume'])) if pd.notna(row['volume']) else None,
                    transactions=int(row['tx_count']) if pd.notna(row.get('tx_count')) else None,
                    active_addresses=int(row['active_addresses']) if pd.notna(row.get('active_addresses')) else None
                )
                db.add(market_data)
                added_count += 1
        
        db.commit()
        logger.info(f"Added {added_count} new records for {symbol}")
        return added_count


from decimal import Decimal