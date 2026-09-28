from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectedFace:
    box: tuple[float, float, float, float]
    landmarks: np.ndarray
    confidence: float
    image: np.ndarray


@dataclass(frozen=True)
class MatchCandidate:
    person_id: int
    similarity: float


class FaceDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> DetectedFace | None: ...


class FaceEmbedder(ABC):
    @abstractmethod
    def embed(self, face: DetectedFace) -> list[float]: ...


class LivenessDetector(ABC):
    @abstractmethod
    def check(self, face: DetectedFace) -> float: ...


class FaceMatcher(ABC):
    @abstractmethod
    def match(self, embedding: list[float]) -> tuple[MatchCandidate | None, MatchCandidate | None]: ...
