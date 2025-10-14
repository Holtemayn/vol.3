from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.core.config import settings

_ENGINE = None
_SESSION_FACTORY = None


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        if not settings.DATABASE_URL:
            raise RuntimeError("DATABASE_URL er ikke sat")
        _ENGINE = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
    return _ENGINE


def get_session() -> Session:
    global _SESSION_FACTORY
    if _SESSION_FACTORY is None:
        engine = get_engine()
        _SESSION_FACTORY = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return _SESSION_FACTORY()
