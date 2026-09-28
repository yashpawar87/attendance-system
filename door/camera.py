import time
from collections.abc import Iterator

import cv2


def frames(camera_index: int = 0, sample_fps: float = 5.0) -> Iterator[bytes]:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")
    interval = 1.0 / sample_fps
    try:
        while True:
            started = time.monotonic()
            ok, frame = capture.read()
            if not ok:
                break
            ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                yield encoded.tobytes()
            time.sleep(max(0.0, interval - (time.monotonic() - started)))
    finally:
        capture.release()


def video_frames(video_path: str, sample_fps: float = 5.0, loop: bool = False) -> Iterator[bytes]:
    """Replay a video as camera-like JPEG frames at approximately sample_fps."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(source_fps / sample_fps))
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                if not loop:
                    break
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_number = 0
                continue
            if frame_number % stride == 0:
                ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    yield encoded.tobytes()
                    time.sleep(1.0 / sample_fps)
            frame_number += 1
    finally:
        capture.release()
