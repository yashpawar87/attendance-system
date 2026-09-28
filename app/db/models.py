import json
from datetime import date, datetime
from typing import Any

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Integer, LargeBinary, String, UniqueConstraint, func
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import UserDefinedType

from app.db.database import Base


class Vector128(UserDefinedType):
    """PostgreSQL VECTOR(128), with a JSON TEXT representation for SQLite tests."""

    cache_ok = True

    def get_col_spec(self, **kw: Any) -> str:
        return "TEXT"

    def bind_processor(self, dialect: Any):
        def process(value: list[float] | None) -> str | None:
            return json.dumps(value) if value is not None else None

        return process

    def result_processor(self, dialect: Any, coltype: Any):
        def process(value: str | list[float] | None) -> list[float] | None:
            if value is None or isinstance(value, list):
                return value
            return json.loads(value)

        return process



@compiles(Vector128, "postgresql")
def compile_vector_postgresql(element, compiler, **kw):
    return "VECTOR(128)"


class Person(Base):
    __tablename__ = "people"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    employee_code: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    profile_photo: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    profile_photo_content_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    embeddings: Mapped[list["FaceEmbedding"]] = relationship(back_populates="person", cascade="all, delete-orphan")


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("people.id", ondelete="CASCADE"), index=True)
    embedding: Mapped[list[float]] = mapped_column(Vector128, nullable=False)
    model_name: Mapped[str] = mapped_column(String(100))
    model_version: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    person: Mapped[Person] = relationship(back_populates="embeddings")


class Door(Base):
    __tablename__ = "doors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    location: Mapped[str | None] = mapped_column(String(255), nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Attendance(Base):
    __tablename__ = "attendance"
    __table_args__ = (UniqueConstraint("person_id", "attendance_date", "event_type", name="uq_attendance_person_day_event"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("people.id", ondelete="CASCADE"), index=True)
    door_id: Mapped[int | None] = mapped_column(ForeignKey("doors.id", ondelete="SET NULL"), nullable=True)
    event_type: Mapped[str] = mapped_column(String(50), default="check_in")
    event_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    attendance_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    similarity: Mapped[float] = mapped_column()
    liveness_score: Mapped[float] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    person: Mapped[Person] = relationship()
    door: Mapped[Door | None] = relationship()
