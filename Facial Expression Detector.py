import cv2
import time
import threading
import mediapipe as mp
from mediapipe.tasks.python.vision import (
    FaceDetector,
    FaceDetectorOptions,
    RunningMode,
)
from mediapipe import Image, ImageFormat
import requests

BaseOptions = mp.tasks.BaseOptions


results_by_ts = {}
results_lock = threading.Lock()

def result_callback(result, output_image, timestamp_ms: int):
    with results_lock:
        results_by_ts[timestamp_ms] = result

MODEL_FILENAME = "face_detection_short_range.tflite"

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=MODEL_FILENAME),
    running_mode=RunningMode.LIVE_STREAM,
    result_callback=result_callback,
)
detector = FaceDetector.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Unable to open webcam")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

      
        ts = int(time.time() * 1000)
        detector.detect_async(mp_image, ts)

       
        detections = None
        wait_start = time.time()
        while time.time() - wait_start < 0.5:  # 500 ms timeout
            with results_lock:
                if ts in results_by_ts:
                    detections = results_by_ts.pop(ts)
                    break
            time.sleep(0.005)

        if detections and detections.detections:
            h, w, _ = frame.shape
            for det in detections.detections:
       
                bbox = det.bounding_box
                x = int(bbox.origin_x * w)
                y = int(bbox.origin_y * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

              
                label = ""
                if det.categories:
                    cat = det.categories[0]
                    score = getattr(cat, "score", None)
                    name = getattr(cat, "category_name", None) or getattr(cat, "label", None)
                    if name is not None and score is not None:
                        label = f"{name}: {score:.2f}"
                    elif score is not None:
                        label = f"{score:.2f}"
                if label:
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Face Detector (MediaPipe Tasks)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    try:
        detector.close()
    except Exception:
        pass