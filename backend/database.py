from sqlalchemy import create_engine, Column, Integer, String, DECIMAL, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import date
import os

# SQLite
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "crypto_forecast.db")
print("DATABASE PATH:", DB_PATH)
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class Cryptocurrency(Base):
    __tablename__ = "cryptocurrencies"
    
    coin_id = Column(Integer, primary_key=True, autoincrement=True)
    coin_name = Column(String(100), nullable=False)
    symbol = Column(String(10), nullable=False)
    
    models = relationship("Model", back_populates="cryptocurrency")
    forecasts = relationship("Forecast", back_populates="cryptocurrency")
    market_data = relationship("MarketData", back_populates="cryptocurrency")


class MarketData(Base):
    __tablename__ = "market_data"
    
    coin_id = Column(Integer, ForeignKey("cryptocurrencies.coin_id"), primary_key=True, nullable=False)
    date = Column(Date, primary_key=True, nullable=False)
    close_price = Column(DECIMAL(16, 8), nullable=False)
    volume = Column(DECIMAL(16, 8))
    active_addresses = Column(Integer)
    transactions = Column(Integer)    
    
    cryptocurrency = relationship("Cryptocurrency", back_populates="market_data")


class Model(Base):
    __tablename__ = "models"
    
    model_id = Column(Integer, primary_key=True, autoincrement=True)
    coin_id = Column(Integer, ForeignKey("cryptocurrencies.coin_id"), nullable=False)
    model_name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)  # xgb_lstm, xgb_tcn, tcn_lstm
    horizon = Column(Integer, nullable=False)  # 30 или 180
    model_path = Column(String(500), nullable=False)  # путь до папки с моделью
    scaler_path = Column(String(500))  # путь до scaler
    created_at = Column(Date, nullable=False, default=date.today)
    lags = Column(Integer)  # Добавить
    lookback = Column(Integer, nullable=False)
    is_active = Column(Integer, default=1)  # 1 - активная, 0 - устаревшая
    
    cryptocurrency = relationship("Cryptocurrency", back_populates="models")
    forecasts = relationship("Forecast", back_populates="model")
    metrics = relationship("ModelMetric", back_populates="model")


class Forecast(Base):
    __tablename__ = "forecasts"
    
    forecast_id = Column(Integer, primary_key=True, autoincrement=True)
    coin_id = Column(Integer, ForeignKey("cryptocurrencies.coin_id"), nullable=False)
    model_id = Column(Integer, ForeignKey("models.model_id"), nullable=False)
    forecast_date = Column(Date, nullable=False)
    target_date = Column(Date, nullable=False)
    predicted_price = Column(DECIMAL(16, 8))
    lower_bound = Column(DECIMAL(16, 8))
    upper_bound = Column(DECIMAL(16, 8))
    
    cryptocurrency = relationship("Cryptocurrency", back_populates="forecasts")
    model = relationship("Model", back_populates="forecasts")


class ModelMetric(Base):
    __tablename__ = "model_metrics"
    
    metric_id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("models.model_id"), nullable=False)
    calculated_at = Column(Date, nullable=False)
    mae = Column(DECIMAL(16, 8))
    mape = Column(DECIMAL(16, 8))
    n_samples = Column(Integer)
    
    model = relationship("Model", back_populates="metrics")


class Admin(Base):
    __tablename__ = "admins"
    
    admin_id = Column(Integer, primary_key=True, autoincrement=True)
    login = Column(String(50), nullable=False, unique=True)
    password = Column(String(255), nullable=False)
    registration_date = Column(Date, nullable=False, default=date.today)


class GarchConfig(Base):
    __tablename__ = "garch_configs"
    
    garch_id = Column(Integer, primary_key=True, autoincrement=True)
    coin_id = Column(Integer, ForeignKey("cryptocurrencies.coin_id"), nullable=False, unique=True)
    model_type = Column(String(20), nullable=False)  # GARCH, GJR-GARCH
    p = Column(Integer, nullable=False)
    q = Column(Integer, nullable=False)
    o = Column(Integer, default=0)  # для GJR-GARCH
    power = Column(Integer, default=2)
    
    cryptocurrency = relationship("Cryptocurrency", back_populates="garch_config")


Cryptocurrency.garch_config = relationship("GarchConfig", back_populates="cryptocurrency", uselist=False)