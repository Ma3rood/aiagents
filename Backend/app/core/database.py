from motor.motor_asyncio import AsyncIOMotorClient
import redis
from app.core.config import settings

# MongoDB Connection
mongodb_client: AsyncIOMotorClient = None

async def connect_to_mongodb():
    global mongodb_client
    try:
        mongodb_client = AsyncIOMotorClient(settings.MONGODB_URL)
        print("Connected to MongoDB.")
    except Exception as e:
        print(f"Could not connect to MongoDB: {e}")
        raise e

async def close_mongodb_connection():
    global mongodb_client
    if mongodb_client:
        mongodb_client.close()
        print("MongoDB connection closed.")

def get_database():
    return mongodb_client[settings.MONGODB_DB_NAME]

# Redis Connection
redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    decode_responses=True
)

def get_redis():
    return redis_client 