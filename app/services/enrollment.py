import cv2
import numpy as np

from app.core.config import Settings
from app.db import repositories
from app.db.models import FaceEmbedding, Person
from app.vision.interfaces import FaceDetector, FaceEmbedder


class EnrollmentService:
    def __init__(self, detector: FaceDetector, embedder: FaceEmbedder, settings: Settings):
        self.detector = detector
        self.embedder = embedder
        self.settings = settings

    def enroll(self, db, person: Person, images: list[np.ndarray]) -> list[FaceEmbedding]:
        if not images:
            raise ValueError("At least one enrollment image is required")
        created: list[FaceEmbedding] = []
        for image in images:
            if hasattr(self.detector, "detect_all"):
                faces = self.detector.detect_all(image)
                if len(faces) != 1:
                    raise ValueError("Each enrollment image must contain exactly one clear face")
                face = faces[0]
            else:
                face = self.detector.detect(image)
                if face is None:
                    raise ValueError("Each enrollment image must contain one clear face")
            x, y, width, height = face.box
            if width < 40 or height < 40 or width * height < image.shape[0] * image.shape[1] * 0.01:
                raise ValueError("Enrollment face is too small")
            face = type(face)(face.box, face.landmarks, face.confidence, image)
            embedding = self.embedder.embed(face)
            if len(embedding) != 128:
                raise ValueError("Enrollment model must return a 128-dimensional embedding")
            created.append(repositories.add_embedding(db, person.id, embedding, "sface", "2021dec"))
        encoded_ok, encoded = cv2.imencode(".jpg", images[0], [cv2.IMWRITE_JPEG_QUALITY, 90])
        if encoded_ok:
            repositories.set_profile_photo(db, person, encoded.tobytes())
        return created
