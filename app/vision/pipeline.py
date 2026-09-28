from dataclasses import dataclass

import numpy as np

from app.core.config import Settings
from app.vision.interfaces import FaceDetector, FaceEmbedder, LivenessDetector


@dataclass(frozen=True)
class RecognitionResult:
    person_id: int
    similarity: float
    liveness: float


@dataclass(frozen=True)
class FaceObservation:
    box: tuple[float, float, float, float]
    detector_confidence: float
    person_id: int | None
    similarity: float | None
    liveness: float
    matched: bool


class RecognitionPipeline:
    def __init__(self, detector: FaceDetector, liveness: LivenessDetector, embedder: FaceEmbedder, settings: Settings):
        self.detector = detector
        self.liveness = liveness
        self.embedder = embedder
        self.settings = settings

    def recognize(
        self,
        image: np.ndarray,
        matcher,
        *,
        enforce_liveness: bool = True,
        enforce_gap: bool = True,
    ) -> RecognitionResult | None:
        observations = self.inspect(image, matcher, enforce_liveness=enforce_liveness, enforce_gap=enforce_gap)
        accepted = [observation for observation in observations if observation.matched]
        if not accepted:
            return None
        best = max(accepted, key=lambda observation: observation.similarity or -1)
        return RecognitionResult(best.person_id, best.similarity or 0.0, best.liveness)

    def inspect(
        self,
        image: np.ndarray,
        matcher,
        *,
        enforce_liveness: bool = True,
        enforce_gap: bool = True,
    ) -> list[FaceObservation]:
        faces = self.detector.detect_all(image) if hasattr(self.detector, "detect_all") else [self.detector.detect(image)]
        observations: list[FaceObservation] = []
        for detected in faces:
            if detected is None:
                continue
            face = type(detected)(detected.box, detected.landmarks, detected.confidence, image)
            live_score = self.liveness.check(face)
            if enforce_liveness and live_score < self.settings.liveness_threshold:
                observations.append(FaceObservation(face.box, face.confidence, None, None, live_score, False))
                continue
            embedding = self.embedder.embed(face)
            best, second = matcher.match(embedding)
            similarity = best.similarity if best else None
            matched = best is not None and best.similarity >= self.settings.recognition_threshold
            if matched and enforce_gap and second is not None and best.similarity - second.similarity < self.settings.similarity_gap_threshold:
                matched = False
            observations.append(
                FaceObservation(
                    face.box,
                    face.confidence,
                    best.person_id if matched and best is not None else None,
                    similarity,
                    live_score,
                    matched,
                )
            )
        return observations
