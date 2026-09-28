from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from app.vision.interfaces import DetectedFace, LivenessDetector


class MiniFASNetLivenessDetector(LivenessDetector):
    def __init__(self, model_path: str | Path, live_class_index: int = 0, crop_scale: float = 2.7):
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        shape = model_input.shape
        self.input_height = int(shape[2]) if isinstance(shape[2], int) else 80
        self.input_width = int(shape[3]) if isinstance(shape[3], int) else 80
        self.live_class_index = live_class_index
        self.crop_scale = crop_scale

    def check(self, face: DetectedFace) -> float:
        crop = self._scaled_crop(face.image, face.box)
        resized = cv2.resize(crop, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(resized, scalefactor=1 / 255.0, size=(self.input_width, self.input_height), swapRB=True)
        logits = np.asarray(self.session.run(None, {self.input_name: blob.astype(np.float32)})[0]).reshape(-1)
        if logits.size == 1:
            return float(1 / (1 + np.exp(-float(logits[0]))))
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= probabilities.sum()
        if self.live_class_index >= probabilities.size:
            raise ValueError("Configured live class index is not present in MiniFASNet output")
        return float(probabilities[self.live_class_index])

    def _scaled_crop(self, image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
        x, y, width, height = box
        center_x, center_y = x + width / 2, y + height / 2
        crop_width, crop_height = width * self.crop_scale, height * self.crop_scale
        left = max(0, int(center_x - crop_width / 2))
        top = max(0, int(center_y - crop_height / 2))
        right = min(image.shape[1], int(center_x + crop_width / 2))
        bottom = min(image.shape[0], int(center_y + crop_height / 2))
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            raise ValueError("Face crop is outside the image")
        return crop
