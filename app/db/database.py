import threading
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


_engine = None
_session_factory = None
_db_lock = threading.Lock()


def setup_db(settings=None):
    global _engine, _session_factory
    if settings is None:
        settings = get_settings()
    with _db_lock:
        if _engine is None:
            _engine = create_engine(settings.database_url, pool_pre_ping=True)
            _session_factory = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)


def get_engine():
    if _engine is None:
        setup_db()
    return _engine


def get_db() -> Generator[Session, None, None]:
    if _session_factory is None:
        setup_db()
    db = _session_factory()
    try:
        yield db
    finally:
        db.close()
