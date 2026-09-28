import requests


class AttendanceApiClient:
    def __init__(self, api_url: str, token: str, source_id: str = "door-camera"):
        self.api_url = api_url.rstrip('/')
        self.headers = {"Authorization": f"Bearer {token}"}
        self.source_id = source_id

    def identify(self, image: bytes, replay_demo: bool = False) -> dict:
        self.publish_frame(image)
        endpoint = "/api/v1/recognition/demo-identify" if replay_demo else "/api/v1/recognition/identify"
        response = requests.post(
            f"{self.api_url}{endpoint}",
            files={"image": ("frame.jpg", image, "image/jpeg")},
            data={"source_id": self.source_id},
            headers=self.headers,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    def publish_frame(self, image: bytes) -> None:
        response = requests.post(
            f"{self.api_url}/api/v1/camera/frame",
            files={"image": ("frame.jpg", image, "image/jpeg")},
            data={"source_id": self.source_id},
            headers=self.headers,
            timeout=10,
        )
        response.raise_for_status()

    def mark(self, result: dict, door_id: int | None = None, replay_demo: bool = False) -> dict:
        payload = {"person_id": result["person_id"], "similarity": result["similarity"], "liveness_score": result["liveness"], "door_id": door_id}
        endpoint = "/api/v1/attendance/demo-mark" if replay_demo else "/api/v1/attendance/mark"
        response = requests.post(f"{self.api_url}{endpoint}", json=payload, headers=self.headers, timeout=10)
        response.raise_for_status()
        return response.json()
