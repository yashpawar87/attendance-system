from collections import defaultdict, deque
from datetime import datetime, timezone
from threading import Lock

import numpy as np

from app.core.config import Settings
from app.vision.interfaces import FaceMatcher
from app.vision.pipeline import FaceObservation, RecognitionPipeline, RecognitionResult


class RecognitionState:
    def __init__(self):
        self.recent: dict[str, deque[tuple[int, datetime]]] = defaultdict(deque)
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
        result = None
        if accepted:
            selected = max(accepted, key=lambda observation: observation.similarity or -1)
            candidate = RecognitionResult(selected.person_id, selected.similarity or 0.0, selected.liveness)
            # A replay is an explicit, operator-triggered demo. Its frames are
            # already coming from a known video, so do not make the demo wait
            # for the normal live-camera temporal confirmation window.
            result = candidate if demo_replay else self._confirm(candidate, source_id)
        else:
            with self._state.lock:
                self._state.recent.pop(source_id, None)
        return result, observations

    def _confirm(self, candidate: RecognitionResult, source_id: str) -> RecognitionResult | None:
        now = datetime.now(timezone.utc)
        with self._state.lock:
            history = self._state.recent[source_id]
            while history and (now - history[0][1]).total_seconds() > self.settings.temporal_window_seconds:
                history.popleft()
            if history and history[-1][0] != candidate.person_id:
                history.clear()
            history.append((candidate.person_id, now))
            if len(history) < self.settings.temporal_match_count:
                return None
            history.clear()
        return candidate
