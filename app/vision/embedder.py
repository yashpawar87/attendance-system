from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from app.vision.interfaces import DetectedFace, FaceEmbedder


REFERENCE_LANDMARKS = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


def align_face(image: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    transform, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), REFERENCE_LANDMARKS, method=cv2.LMEDS)
    if transform is None:
        raise ValueError("Could not estimate face alignment transform")
    return cv2.warpAffine(image, transform, (size, size), borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class SFaceEmbedder(FaceEmbedder):
    def __init__(self, model_path: str | Path, mean: tuple[float, float, float] = (127.5, 127.5, 127.5), scale: float = 1 / 128.0, swap_rb: bool = True):
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.mean = mean
        self.scale = scale
        self.swap_rb = swap_rb

    def embed(self, face: DetectedFace) -> list[float]:
        aligned = align_face(face.image, face.landmarks)
        blob = cv2.dnn.blobFromImage(aligned, scalefactor=self.scale, size=(112, 112), mean=self.mean, swapRB=self.swap_rb)
        output = np.asarray(self.session.run(None, {self.input_name: blob.astype(np.float32)})[0]).reshape(-1).astype(np.float32)
        if output.size != 128:
            raise ValueError(f"SFace returned {output.size} values; expected 128")
        norm = np.linalg.norm(output)
        if norm == 0:
            raise ValueError("SFace returned a zero embedding")
        return (output / norm).tolist()
