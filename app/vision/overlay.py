import cv2
import numpy as np

from app.vision.pipeline import FaceObservation


def draw_face_overlays(image: np.ndarray, observations: list[FaceObservation]) -> np.ndarray:
    """Draw green known-person boxes and red unknown/spoof boxes in memory."""
    annotated = image.copy()
    for observation in observations:
        x, y, width, height = observation.box
        left = max(0, int(x))
        top = max(0, int(y))
        right = min(annotated.shape[1] - 1, int(x + width))
        bottom = min(annotated.shape[0] - 1, int(y + height))
        if right <= left or bottom <= top:
            continue
        known = observation.matched and observation.person_id is not None
        color = (66, 180, 105) if known else (55, 70, 225)
        label = f"Employee #{observation.person_id}" if known else "Unknown / intruder"
        cv2.rectangle(annotated, (left, top), (right, bottom), color, 3)
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        label_top = max(0, top - text_height - baseline - 10)
        cv2.rectangle(annotated, (left, label_top), (left + text_width + 14, top), color, -1)
        cv2.putText(annotated, label, (left + 7, max(text_height + 2, top - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated
