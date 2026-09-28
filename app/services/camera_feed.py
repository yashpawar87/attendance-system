from collections.abc import Iterator
from threading import Condition


class CameraFrameStore:
    """Short-lived in-memory relay; frames are never written to disk or logs."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._frames: dict[str, tuple[int, bytes]] = {}
        self._intruder_stats: dict[str, dict[str, int | bool]] = {}
        self._sequence = 0

    def publish(self, source_id: str, frame: bytes) -> None:
        with self._condition:
            self._sequence += 1
            self._frames[source_id] = (self._sequence, frame)
            self._condition.notify_all()

    def record_intruders(self, source_id: str, unknown_faces: int) -> None:
        """Count a new unknown-person appearance, not every repeated frame."""
        with self._condition:
            stats = self._intruder_stats.setdefault(source_id, {"intruder_events": 0, "unknown_face_detections": 0, "active": False})
            stats["unknown_face_detections"] = int(stats["unknown_face_detections"]) + max(0, unknown_faces)
            currently_active = unknown_faces > 0
            if currently_active and not bool(stats["active"]):
                stats["intruder_events"] = int(stats["intruder_events"]) + 1
            stats["active"] = currently_active

    def status(self, source_id: str) -> dict[str, int | str]:
        with self._condition:
            stats = self._intruder_stats.get(source_id, {"intruder_events": 0, "unknown_face_detections": 0})
            return {
                "source_id": source_id,
                "intruder_events": int(stats.get("intruder_events", 0)),
                "unknown_face_detections": int(stats.get("unknown_face_detections", 0)),
            }

    def stream(self, source_id: str) -> Iterator[bytes]:
        last_sequence = -1
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: source_id in self._frames and self._frames[source_id][0] != last_sequence,
                    timeout=5.0,
                )
                current = self._frames.get(source_id)
            if current is None:
                continue
            last_sequence, frame = current
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                + frame
                + b"\r\n"
            )
