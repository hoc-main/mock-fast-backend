from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text
from dotenv import load_dotenv
import logging
import os

load_dotenv()
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://root:root@localhost:5432/hp"
)

engine = create_async_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,

    # Sends "SELECT 1" before giving a connection to your code.
    # Detects and discards dead connections automatically.
    pool_pre_ping=True,

    # Recycle connections every 10 min — prevents using connections
    # that Postgres has already closed on its side (hosted DBs like
    # Supabase/RDS often have a 10min idle timeout).
    pool_recycle=600,

    # Raise after 30s if no connection available — avoids hanging forever.
    pool_timeout=30,

    # asyncpg-level timeouts — kills the coroutine if the server is
    # unresponsive rather than blocking indefinitely.
    connect_args={
        "command_timeout": 60,
        "timeout": 10,
    },

    # echo=(os.getenv("APP_ENV") == "development"),
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db():
    """
    FastAPI dependency — yields one async session per request.
    Rolls back on any exception so a failed request never leaves
    a broken transaction open in the pool.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_db_connection() -> bool:
    """
    Health-check helper — called at startup and by /health endpoint.
    Returns True if the DB is reachable, False otherwise.
    """
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"DB connection check failed: {e}")
        return False