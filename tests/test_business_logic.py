from datetime import datetime, timezone

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.db.database import Base
from app.db.models import Attendance, FaceEmbedding, Person
from app.db.repositories import find_best_matches, insert_attendance
from app.services.attendance_service import AttendanceService
from app.services.recognition_service import RecognitionService
from app.vision.interfaces import DetectedFace, MatchCandidate
from app.vision.pipeline import FaceObservation, RecognitionPipeline


def make_db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_pgvector_match_fallback_returns_best_person_and_respects_multiple_embeddings():
    db = make_db()
    person = Person(employee_code="E1", name="A", active=True)
    other = Person(employee_code="E2", name="B", active=True)
    db.add_all([person, other])
    db.commit()
    db.add_all([
        FaceEmbedding(person_id=person.id, embedding=[1.0, 0.0], model_name="sface", model_version="1"),
        FaceEmbedding(person_id=person.id, embedding=[0.8, 0.2], model_name="sface", model_version="1"),
        FaceEmbedding(person_id=other.id, embedding=[0.0, 1.0], model_name="sface", model_version="1"),
    ])
    db.commit()
    matches = find_best_matches(db, [1.0, 0.0])
    assert matches[0][0] == person.id
    assert matches[0][1] > matches[1][1]


def test_attendance_unique_constraint_is_the_duplicate_guard():
    db = make_db()
    event_at = datetime(2026, 9, 25, 3, 30, tzinfo=timezone.utc)
    first, inserted = insert_attendance(db, person_id=1, door_id=None, event_type="check_in", event_at=event_at, attendance_date=event_at.date(), similarity=.9, liveness_score=.95)
    assert inserted is True
    second, inserted = insert_attendance(db, person_id=1, door_id=None, event_type="check_in", event_at=event_at, attendance_date=event_at.date(), similarity=.8, liveness_score=.9)
    assert inserted is False
    assert second.id == first.id


def test_attendance_date_uses_configured_local_timezone():
    settings = Settings(attendance_timezone="Asia/Kolkata")
    service = AttendanceService(settings)
    db = make_db()
    row, _ = service.mark(db, person_id=1, similarity=.9, liveness_score=.9, event_at=datetime(2026, 9, 24, 19, 0, tzinfo=timezone.utc))
    assert row.attendance_date.isoformat() == "2026-09-25"


class FakeDetector:
    def detect(self, image):
        return DetectedFace((0, 0, 100, 100), np.zeros((5, 2), dtype=np.float32), .99, image)


class FakeLiveness:
    def check(self, face):
        return .95


class FakeEmbedder:
    def embed(self, face):
        return [1.0] + [0.0] * 127


class FakeMatcher:
    def match(self, embedding):
        return MatchCandidate(7, .91), MatchCandidate(8, .2)


def test_pipeline_enforces_liveness_threshold_and_similarity_gap():
    settings = Settings(temporal_match_count=1, recognition_threshold=.8, similarity_gap_threshold=.1, liveness_threshold=.7)
    pipeline = RecognitionPipeline(FakeDetector(), FakeLiveness(), FakeEmbedder(), settings)
    result = pipeline.recognize(np.zeros((120, 120, 3), dtype=np.uint8), FakeMatcher())
    assert result is not None
    assert result.person_id == 7


def test_temporal_confirmation_requires_consecutive_matches():
    settings = Settings(temporal_match_count=3, temporal_window_seconds=1)
    pipeline = RecognitionPipeline(FakeDetector(), FakeLiveness(), FakeEmbedder(), settings)
    service = RecognitionService(pipeline, lambda: FakeMatcher(), settings)
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    assert service.identify(frame) is None
    assert service.identify(frame) is None
    assert service.identify(frame) is not None


def test_demo_replay_returns_match_without_temporal_confirmation():
    settings = Settings(temporal_match_count=3, temporal_window_seconds=1)
    pipeline = RecognitionPipeline(FakeDetector(), FakeLiveness(), FakeEmbedder(), settings)
    service = RecognitionService(pipeline, lambda: FakeMatcher(), settings)
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    result, observations = service.identify_with_observations(frame, source_id="demo", demo_replay=True)
    assert result is not None
    assert result.person_id == 7
    assert observations[0].matched is True


def test_pipeline_deduplicates_overlapping_observations_for_one_person():
    observations = [
        FaceObservation((10, 10, 100, 100), .90, 7, .91, .95, True),
        FaceObservation((12, 12, 98, 98), .80, 7, .88, .95, True),
        FaceObservation((220, 10, 100, 100), .85, 8, .90, .95, True),
    ]
    unique = RecognitionPipeline._deduplicate_observations(observations)
    assert [observation.person_id for observation in unique] == [7, 8]
    assert unique[0].similarity == .91


def test_demo_replay_rejects_ambiguous_identity_instead_of_picking_wrong_employee():
    class AmbiguousMatcher:
        def match(self, embedding):
            return MatchCandidate(7, .91), MatchCandidate(8, .90)

    settings = Settings(temporal_match_count=1, similarity_gap_threshold=.03, demo_similarity_gap_threshold=.03)
    pipeline = RecognitionPipeline(FakeDetector(), FakeLiveness(), FakeEmbedder(), settings)
    service = RecognitionService(pipeline, lambda: AmbiguousMatcher(), settings)
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    result, observations = service.identify_with_observations(frame, source_id="ambiguous-demo", demo_replay=True)
    assert result is None
    assert observations[0].matched is False
    assert observations[0].person_id is None
