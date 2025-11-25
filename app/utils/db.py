from __future__ import annotations

import logging
from typing import AsyncGenerator, Optional

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel, select

from app.utils.models import User
from config import DATABASE_URL, ENV

logger = logging.getLogger(__name__)

engine_kwargs = {
    "echo": ENV == "dev",
    "pool_pre_ping": ENV != "dev",
}

# SQLite-specific connect_args
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs.update({"connect_args": {"check_same_thread": False}})

engine = create_async_engine(DATABASE_URL, **engine_kwargs)

# Log resolved DB URL
logger.info("DATABASE_URL resolved to: %s", str(engine.url))

# Enable foreign key support for SQLite
if DATABASE_URL.startswith("sqlite"):

    @event.listens_for(engine.sync_engine, "connect")
    def _sqlite_enable_foreign_keys(dbapi_connection, connection_record) -> None:  # type: ignore[override]
        try:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
        except Exception:
            logger.exception("Could not enable SQLite foreign keys PRAGMA")


# Async session factory
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI routes:

        async def route(session: AsyncSession = Depends(async_session)):
            ...

    Or manual usage:

        async with async_session() as session:
            ...
    """
    session: AsyncSession = AsyncSessionLocal()
    try:
        yield session
    finally:
        await session.close()


async def init_db() -> None:
    """
    Initialize database tables (for dev / first deployment).
    In real production, use Alembic migrations instead of this.
    """
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    logger.info("Database initialized (tables created if missing).")


async def get_user_by_firebase_uid(
    session: AsyncSession, firebase_uid: str
) -> Optional[User]:
    stmt = select(User).where(User.firebase_uid == firebase_uid)
    result = await session.exec(stmt)
    return result.scalar_one_or_none()


async def get_user_by_id(session: AsyncSession, user_id: str) -> Optional[User]:
    stmt = select(User).where(User.id == user_id)
    result = await session.exec(stmt)
    return result.scalar_one_or_none()
