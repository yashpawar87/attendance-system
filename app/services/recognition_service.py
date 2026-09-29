from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock

import numpy as np

from app.core.config import Settings
from app.vision.interfaces import FaceMatcher
from app.vision.pipeline import FaceObservation, RecognitionPipeline, RecognitionResult


@dataclass
class _FaceTrack:
    recent: deque[tuple[int, tuple[float, float, float, float], datetime]] = field(default_factory=deque)
    confirmed_person_id: int | None = None
    confirmed_box: tuple[float, float, float, float] | None = None
    confirmed_at: datetime | None = None


class RecognitionState:
    def __init__(self):
        # A camera frame can contain more than one person. Keep independent
        # temporal tracks per source instead of allowing the highest-scoring
        # face to overwrite every other face's confirmation state.
        self.tracks: dict[str, list[_FaceTrack]] = defaultdict(list)
        self.lock = Lock()


class RecognitionService:
    def __init__(self, pipeline: RecognitionPipeline, matcher_factory, settings: Settings, state: RecognitionState | None = None):
        self.pipeline = pipeline
        self.matcher_factory = matcher_factory
        self.settings = settings
        self._state = state or RecognitionState()

    def identify(self, image: np.ndarray, source_id: str = "default", *, demo_replay: bool = False) -> RecognitionResult | None:
        result, _ = self.identify_with_observations(image, source_id=source_id, demo_replay=demo_replay)
        return result

    def identify_with_observations(
        self,
        image: np.ndarray,
        source_id: str = "default",
        *,
        demo_replay: bool = False,
    ) -> tuple[RecognitionResult | None, list[FaceObservation]]:
        results, observations = self.identify_many_with_observations(
            image,
            source_id=source_id,
            demo_replay=demo_replay,
        )
        result = max(results, key=lambda item: item.similarity) if results else None
        return result, observations

    def identify_many_with_observations(
        self,
        image: np.ndarray,
        source_id: str = "default",
        *,
        demo_replay: bool = False,
    ) -> tuple[list[RecognitionResult], list[FaceObservation]]:
        gap_threshold = self.settings.demo_similarity_gap_threshold if demo_replay else self.settings.similarity_gap_threshold
        observations = self.pipeline.inspect(
            image,
            self.matcher_factory(),
            enforce_liveness=not demo_replay,
            # Replay mode is an explicit presentation flow. Its synthetic
            # subjects are visually similar, so use the configured demo gap
            # (zero by default) while keeping the stricter live-camera gap.
            enforce_gap=gap_threshold > 0,
            similarity_gap_threshold=gap_threshold,
        )
        accepted = [observation for observation in observations if observation.matched and observation.person_id is not None]
        results: list[RecognitionResult] = []
        if accepted:
            # Confirm every face independently. A frame-level "best match"
            # would silently drop all other employees when several people are
            # visible at once.
            for observation in accepted:
                candidate = RecognitionResult(observation.person_id, observation.similarity or 0.0, observation.liveness)
                confirmed = self._confirm(
                    candidate,
                    source_id,
                    observation.box,
                    required_count=self.settings.demo_temporal_match_count if demo_replay else self.settings.temporal_match_count,
                    window_seconds=self.settings.demo_temporal_window_seconds if demo_replay else self.settings.temporal_window_seconds,
                    lock_track=demo_replay,
                )
                if confirmed is not None:
                    results.append(confirmed)
        else:
            with self._state.lock:
                self._state.tracks.pop(source_id, None)
        return results, observations

    @staticmethod
    def _same_face_track(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> bool:
        first_x, first_y, first_width, first_height = first
        second_x, second_y, second_width, second_height = second
        first_center = (first_x + first_width / 2, first_y + first_height / 2)
        second_center = (second_x + second_width / 2, second_y + second_height / 2)
        center_distance = ((first_center[0] - second_center[0]) ** 2 + (first_center[1] - second_center[1]) ** 2) ** 0.5
        return center_distance <= 1.5 * max(first_width, first_height, second_width, second_height)

    @staticmethod
    def _intersection_over_union(first: tuple[float, float, float, float] | None, second: tuple[float, float, float, float]) -> float:
        if first is None:
            return 0.0
        first_x, first_y, first_width, first_height = first
        second_x, second_y, second_width, second_height = second
        first_right, first_bottom = first_x + first_width, first_y + first_height
        second_right, second_bottom = second_x + second_width, second_y + second_height
        intersection_width = max(0.0, min(first_right, second_right) - max(first_x, second_x))
        intersection_height = max(0.0, min(first_bottom, second_bottom) - max(first_y, second_y))
        intersection = intersection_width * intersection_height
        union = first_width * first_height + second_width * second_height - intersection
        return intersection / union if union > 0 else 0.0

    def _confirm(
        self,
        candidate: RecognitionResult,
        source_id: str,
        box: tuple[float, float, float, float],
        *,
        required_count: int,
        window_seconds: float,
        lock_track: bool = False,
    ) -> RecognitionResult | None:
        now = datetime.now(timezone.utc)
        with self._state.lock:
            tracks = self._state.tracks[source_id]
            tracks[:] = [track for track in tracks if self._track_is_recent(track, now, window_seconds)]
            track = self._find_track(tracks, box, candidate.person_id)
            if track is None:
                track = _FaceTrack()
                tracks.append(track)

            if track.confirmed_person_id is not None and lock_track:
                if track.confirmed_person_id != candidate.person_id:
                    # A different identity with little box overlap can be a
                    # second person entering beside the locked face, not an
                    # identity switch. Start an independent track in that
                    # case; keep rejecting true same-box relabels.
                    if self._intersection_over_union(track.confirmed_box, box) < 0.15:
                        track = _FaceTrack()
                        tracks.append(track)
                    else:
                        # Do not let an unstable embedding relabel one
                        # tracked face as another employee after it was
                        # confirmed.
                        return None
                if track.confirmed_person_id is not None:
                    track.confirmed_box = box
                    track.confirmed_at = now
                    return candidate

            history = track.recent
            while history and (now - history[0][2]).total_seconds() > window_seconds:
                history.popleft()
            if history and (
                history[-1][0] != candidate.person_id
                or not self._same_face_track(history[-1][1], box)
            ):
                history.clear()
            history.append((candidate.person_id, box, now))
            if len(history) < required_count:
                return None
            history.clear()
            if lock_track:
                track.confirmed_person_id = candidate.person_id
                track.confirmed_box = box
                track.confirmed_at = now
        return candidate

    @classmethod
    def _find_track(
        cls,
        tracks: list[_FaceTrack],
        box: tuple[float, float, float, float],
        person_id: int,
    ) -> _FaceTrack | None:
        preferred = []
        fallback = []
        for track in tracks:
            reference = track.confirmed_box or (track.recent[-1][1] if track.recent else None)
            if reference is not None and cls._same_face_track(reference, box):
                first_x, first_y, first_width, first_height = reference
                second_x, second_y, second_width, second_height = box
                first_center = (first_x + first_width / 2, first_y + first_height / 2)
                second_center = (second_x + second_width / 2, second_y + second_height / 2)
                distance = ((first_center[0] - second_center[0]) ** 2 + (first_center[1] - second_center[1]) ** 2) ** 0.5
                known_person_id = track.confirmed_person_id or (track.recent[-1][0] if track.recent else None)
                (preferred if known_person_id == person_id else fallback).append((distance, track))
        candidates = preferred or fallback
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    @staticmethod
    def _track_is_recent(track: _FaceTrack, now: datetime, window_seconds: float) -> bool:
        timestamp = track.confirmed_at or (track.recent[-1][2] if track.recent else None)
        return timestamp is not None and (now - timestamp).total_seconds() <= window_seconds
