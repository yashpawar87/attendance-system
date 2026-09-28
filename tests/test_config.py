import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_thresholds_are_configurable():
    settings = Settings(recognition_threshold=.81, liveness_threshold=.88)
    assert settings.recognition_threshold == .81
    assert settings.liveness_threshold == .88


def test_invalid_timezone_is_rejected():
    with pytest.raises(ValidationError):
        Settings(attendance_timezone="Not/AZone")
