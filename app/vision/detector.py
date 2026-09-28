from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from app.vision.interfaces import DetectedFace, FaceDetector


class YuNetFaceDetector(FaceDetector):
    """YuNet ONNX adapter using ONNX Runtime and explicit post-processing."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.70,
        nms_threshold: float = 0.30,
        top_k: int = 5000,
    ):
        options = ort.SessionOptions()
        # Some older exports list initializers as graph inputs. This is a
        # harmless optimization warning, not an inference failure.
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.input_height = int(shape[2]) if isinstance(shape[2], int) else 640
        self.input_width = int(shape[3]) if isinstance(shape[3], int) else 640
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.strides = (8, 16, 32)

    def detect(self, image: np.ndarray) -> DetectedFace | None:
        candidates = self.detect_all(image)
        return max(candidates, key=lambda item: item.confidence) if candidates else None

    def detect_all(self, image: np.ndarray) -> list[DetectedFace]:
        if image is None or image.size == 0:
            return []
        original_height, original_width = image.shape[:2]
        resized = cv2.resize(image, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        # This YuNet export expects raw uint8-range BGR input. Normalizing or
        # swapping channels collapses its objectness scores.
        blob = cv2.dnn.blobFromImage(resized, scalefactor=1.0, size=(self.input_width, self.input_height), swapRB=False)
        outputs = self.session.run(None, {self.input_name: blob.astype(np.float32)})
        candidates = self._decode_multi_output(outputs, original_width, original_height)
        return candidates if candidates else self._decode_flat_output(outputs, original_width, original_height)

    def _decode_multi_output(self, outputs: list[np.ndarray], width: int, height: int) -> list[DetectedFace]:
        if len(outputs) != 12:
            return []
        faces: list[DetectedFace] = []
        boxes: list[list[int]] = []
        scores: list[float] = []
        scale_x = width / self.input_width
        scale_y = height / self.input_height
        for level, stride in enumerate(self.strides):
            cols = self.input_width // stride
            rows = self.input_height // stride
            cls = np.asarray(outputs[level]).reshape(-1)
            obj = np.asarray(outputs[level + 3]).reshape(-1)
            bbox = np.asarray(outputs[level + 6]).reshape(-1, 4)
            kps = np.asarray(outputs[level + 9]).reshape(-1, 10)
            expected = rows * cols
            if cls.size != expected or obj.size != expected or bbox.shape[0] != expected or kps.shape[0] != expected:
                return []
            for row in range(rows):
                for col in range(cols):
                    index = row * cols + col
                    cls_score = float(np.clip(cls[index], 0.0, 1.0))
                    objectness = float(np.clip(obj[index], 0.0, 1.0))
                    score = float(np.sqrt(cls_score * objectness))
                    if score < self.confidence_threshold:
                        continue
                    values = bbox[index]
                    center_x = (col + float(values[0])) * stride
                    center_y = (row + float(values[1])) * stride
                    box_width = float(np.exp(values[2]) * stride)
                    box_height = float(np.exp(values[3]) * stride)
                    x = (center_x - box_width / 2.0) * scale_x
                    y = (center_y - box_height / 2.0) * scale_y
                    scaled_width = box_width * scale_x
                    scaled_height = box_height * scale_y
                    boxes.append([int(x), int(y), int(scaled_width), int(scaled_height)])
                    scores.append(score)
                    landmarks = np.asarray(
                        [
                            ((kps[index, 2 * point] + col) * stride * scale_x,
                             (kps[index, 2 * point + 1] + row) * stride * scale_y)
                            for point in range(5)
                        ],
                        dtype=np.float32,
                    )
                    faces.append(DetectedFace((x, y, scaled_width, scaled_height), landmarks, score, np.empty((0, 0, 3), dtype=np.uint8)))
        keep = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, self.nms_threshold, top_k=self.top_k)
        indices = np.asarray(keep).reshape(-1).tolist() if len(keep) else []
        return [faces[int(index)] for index in indices]

    def _decode_flat_output(self, outputs: list[np.ndarray], width: int, height: int) -> list[DetectedFace]:
        """Compatibility path for exports that already return [1, N, 15]."""
        rows: list[np.ndarray] = []
        for output in outputs:
            array = np.asarray(output)
            if array.ndim >= 2 and array.shape[-1] >= 15:
                rows.extend(array.reshape(-1, array.shape[-1]))
        scale_x = width / self.input_width
        scale_y = height / self.input_height
        decoded: list[DetectedFace] = []
        for row in rows:
            score = float(row[14])
            if score < self.confidence_threshold:
                continue
            x, y, box_width, box_height = (float(value) for value in row[:4])
            landmarks = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
            decoded.append(
                DetectedFace(
                    (x * scale_x, y * scale_y, box_width * scale_x, box_height * scale_y),
                    landmarks * np.array([scale_x, scale_y], dtype=np.float32),
                    score,
                    np.empty((0, 0, 3), dtype=np.uint8),
                )
            )
        return decoded
