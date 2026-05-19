import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Hỗ trợ cả 2 tên env var: MONGODB_URI (recommend_api.py) và MONGO_URI (cũ)
MONGO_URI = (
    os.getenv("MONGODB_URI")        # ưu tiên — nhất quán với recommend_api.py
    or os.getenv("MONGO_URI")
    or "mongodb://localhost:27017"
)
DB_NAME = (
    os.getenv("MONGODB_DB")         # ưu tiên — nhất quán với recommend_api.py
    or os.getenv("DB_NAME")
    or "tiki_recommendation"
)

_client = None

def get_db():
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI)
    return _client[DB_NAME]

def close_db():
    global _client
    if _client:
        _client.close()
        _client = None