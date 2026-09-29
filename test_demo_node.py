import cv2
import requests
video = cv2.VideoCapture("create_syn_data/P1E_S1_C1/chokepoint_P1E_S1_C1.mp4")
video.set(cv2.CAP_PROP_POS_FRAMES, 180)
ret, frame = video.read()
if ret:
    ret, jpeg = cv2.imencode(".jpg", frame)
    files = {"image": ("demo-frame.jpg", jpeg.tobytes(), "image/jpeg")}
    data = {"source_id": "door-camera"}
    response = requests.post("http://localhost:3000/browser/demo-identify", files=files, data=data)
    print("Status code:", response.status_code)
    print(response.text)
