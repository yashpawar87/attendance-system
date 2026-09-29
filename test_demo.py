import cv2
import requests
video = cv2.VideoCapture("create_syn_data/P1E_S1_C1/chokepoint_P1E_S1_C1.mp4")
found = False
for f in range(0, 1000, 30):
    video.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret, frame = video.read()
    if not ret: break
    ret, jpeg = cv2.imencode(".jpg", frame)
    files = {"image": ("frame.jpg", jpeg.tobytes(), "image/jpeg")}
    response = requests.post("http://localhost:8000/api/v1/recognition/demo-identify", files=files, headers={"Authorization": "Bearer change-me"})
    if response.json().get("detections"):
        print(f"Face found at frame {f}: {response.json()}")
        found = True
        break
if not found:
    print("No faces found in 1000 frames!")
