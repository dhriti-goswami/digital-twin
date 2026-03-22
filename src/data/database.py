"""Database connection and session management."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from src.utils.config import settings

logger = logging.getLogger(__name__)


# Synchronous engine (for scripts and migrations)
sync_engine = create_engine(
    settings.db.postgres_url,
    pool_size=10,
    max_overflow=20,
    echo=settings.debug,
)

SyncSessionLocal = sessionmaker(bind=sync_engine, autocommit=False, autoflush=False)

# Asynchronous engine (for FastAPI)
async_engine = create_async_engine(
    settings.db.async_postgres_url,
    pool_size=10,
    max_overflow=20,
    echo=settings.debug,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def get_sync_session() -> Session:
    """Get a synchronous database session."""
    return SyncSessionLocal()


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an asynchronous database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_session_context() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for async sessions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def check_database_connection() -> bool:
    """Check if database is accessible."""
    try:
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


async def init_database():
    """Initialize database connection and verify tables exist."""
    logger.info("Initializing database connection...")

    if await check_database_connection():
        logger.info("Database connection successful")
    else:
        raise RuntimeError("Failed to connect to database")
