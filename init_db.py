from backend.database import Base, engine, SessionLocal
from backend.database import Cryptocurrency, Admin

# Создаем таблицы
Base.metadata.create_all(bind=engine)

db = SessionLocal()

# ========== Добавляем криптовалюты ==========
cryptos = [
    {"coin_name": "Bitcoin", "symbol": "BTC"},
    {"coin_name": "Ethereum", "symbol": "ETH"},
    {"coin_name": "Ripple", "symbol": "XRP"},
    {"coin_name": "Litecoin", "symbol": "LTC"},
]

for crypto in cryptos:
    exists = db.query(Cryptocurrency).filter(Cryptocurrency.symbol == crypto["symbol"]).first()
    if not exists:
        db.add(Cryptocurrency(**crypto))

db.commit()

# ========== Добавляем админа ==========
admin = db.query(Admin).filter(Admin.login == "admin").first()
if not admin:
    db.add(Admin(login="admin", password="admin123"))

db.commit()

print("=" * 50)
print("Database initialized successfully!")
print(f"Cryptocurrencies: {db.query(Cryptocurrency).count()}")
print(f"Admin: {db.query(Admin).count()}")
print("=" * 50)
print("\n⚠️ Модели не добавлены. Запустите register_models.py")
print("=" * 50)

db.close()