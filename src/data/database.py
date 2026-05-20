"""Database connection and session management."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy import create_engine, text, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from src.utils.config import settings

logger = logging.getLogger(__name__)

# Ensure data directory exists for SQLite
if settings.db.use_sqlite:
    db_path = Path(settings.db.sqlite_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using SQLite database: {db_path}")

# Synchronous engine (for scripts and migrations)
if settings.db.use_sqlite:
    sync_engine = create_engine(
        settings.db.database_url,
        connect_args={"check_same_thread": False},  # Required for SQLite
        echo=settings.debug,
    )
else:
    sync_engine = create_engine(
        settings.db.database_url,
        pool_size=10,
        max_overflow=20,
        echo=settings.debug,
    )

SyncSessionLocal = sessionmaker(bind=sync_engine, autocommit=False, autoflush=False)

# Asynchronous engine (for FastAPI)
if settings.db.use_sqlite:
    async_engine = create_async_engine(
        settings.db.async_database_url,
        echo=settings.debug,
    )
else:
    async_engine = create_async_engine(
        settings.db.async_database_url,
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
    """Initialize database connection and create tables if needed."""
    from src.data.models import Base

    logger.info("Initializing database...")

    # Create tables if they don't exist
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database initialized successfully")
