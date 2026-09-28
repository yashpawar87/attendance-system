# Model weights

Place the three committed ONNX weights in this directory:

- `face_detection_yunet_2023mar.onnx`
- `face_recognition_sface_2021dec.onnx`
- `minifasnet_v2.onnx`

The application refuses recognition requests with HTTP 503 until all three
files are present. Model binaries are intentionally not generated or replaced
with a different inference backend.
