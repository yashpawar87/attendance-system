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
        similarity_gap_threshold: float | None = None,
    ) -> RecognitionResult | None:
        observations = self.inspect(
            image,
            matcher,
            enforce_liveness=enforce_liveness,
            enforce_gap=enforce_gap,
            similarity_gap_threshold=similarity_gap_threshold,
        )
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
        similarity_gap_threshold: float | None = None,
    ) -> list[FaceObservation]:
        gap_threshold = self.settings.similarity_gap_threshold if similarity_gap_threshold is None else similarity_gap_threshold
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
            if matched and enforce_gap and second is not None and best.similarity - second.similarity < gap_threshold:
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
        return self._deduplicate_observations(observations)

    @staticmethod
    def _intersection_over_union(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
        first_x, first_y, first_width, first_height = first
        second_x, second_y, second_width, second_height = second
        first_right, first_bottom = first_x + first_width, first_y + first_height
        second_right, second_bottom = second_x + second_width, second_y + second_height
        intersection_width = max(0.0, min(first_right, second_right) - max(first_x, second_x))
        intersection_height = max(0.0, min(first_bottom, second_bottom) - max(first_y, second_y))
        intersection = intersection_width * intersection_height
        first_area = max(0.0, first_width) * max(0.0, first_height)
        second_area = max(0.0, second_width) * max(0.0, second_height)
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0

    @classmethod
    def _deduplicate_observations(cls, observations: list[FaceObservation]) -> list[FaceObservation]:
        """Collapse overlapping boxes for the same recognized face in a frame."""
        unique: list[FaceObservation] = []
        for observation in observations:
            duplicate_index = None
            for index, existing in enumerate(unique):
                same_person = observation.person_id is not None and observation.person_id == existing.person_id
                both_unknown = observation.person_id is None and existing.person_id is None
                overlap_threshold = 0.35 if same_person else 0.50
                if (same_person or both_unknown) and cls._intersection_over_union(observation.box, existing.box) >= overlap_threshold:
                    duplicate_index = index
                    break
            if duplicate_index is None:
                unique.append(observation)
                continue
            existing = unique[duplicate_index]
            observation_quality = (observation.matched, observation.similarity or -1.0, observation.detector_confidence)
            existing_quality = (existing.matched, existing.similarity or -1.0, existing.detector_confidence)
            if observation_quality > existing_quality:
                unique[duplicate_index] = observation
        return unique
