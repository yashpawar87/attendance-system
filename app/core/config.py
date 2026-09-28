from functools import lru_cache
from zoneinfo import ZoneInfo

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Door Attendance API"
    database_url: str = "postgresql+psycopg://attendance:attendance@localhost:5432/attendance"
    recognition_threshold: float = Field(0.55, ge=-1, le=1)
    similarity_gap_threshold: float = Field(0.03, ge=0, le=1)
    liveness_threshold: float = Field(0.70, ge=0, le=1)
    # MiniFASNetV2 exports three probabilities in this order:
    # live, print attack, replay attack.
    liveness_live_class_index: int = Field(0, ge=0, le=2)
    # Disabled by default. When enabled, the explicitly named replay-demo
    # endpoints can populate demo_replay attendance rows from an MP4.
    demo_replay_enabled: bool = False
    attendance_timezone: str = "Asia/Kolkata"
    temporal_match_count: int = Field(3, ge=1, le=20)
    temporal_window_seconds: float = Field(1.0, gt=0, le=60)
    attendance_event_type: str = "check_in"
    model_dir: str = "app/vision/models"
    detection_model: str = "face_detection_yunet_2023mar.onnx"
    embedding_model: str = "face_recognition_sface_2021dec.onnx"
    liveness_model: str = "minifasnet_v2.onnx"
    api_token: str = "change-me"
    cors_origins: str = "http://localhost:3000"
    max_upload_bytes: int = Field(8 * 1024 * 1024, gt=0)

    @field_validator("database_url", mode="before")
    @classmethod
    def normalize_database_url(cls, value: str) -> str:
        value = str(value)
        if value.startswith("postgres://"):
            return "postgresql+psycopg://" + value.removeprefix("postgres://")
        if value.startswith("postgresql://"):
            return "postgresql+psycopg://" + value.removeprefix("postgresql://")
        return value

    @field_validator("attendance_timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except Exception as exc:
            raise ValueError(f"Unknown timezone: {value}") from exc
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
