from typing import List, Tuple

import cv2
import numpy as np

# Try to import mediapipe; fall back to OpenCV Haar cascades if unavailable.
try:
    import mediapipe as mp  # type: ignore
    _HAVE_MEDIAPIPE = True
except Exception:
    mp = None  # type: ignore
    _HAVE_MEDIAPIPE = False


def _clip_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def detect_faces(image_bgr: np.ndarray, min_confidence: float = 0.6) -> List[Tuple[Tuple[int, int, int, int], float]]:
    """Detect faces using MediaPipe if available; otherwise OpenCV Haar cascade.

    Returns a list of tuples: ((x1, y1, x2, y2), score)
    Coordinates are absolute pixels in the input image space.
    """
    h, w = image_bgr.shape[:2]

    # Preferred: MediaPipe
    if _HAVE_MEDIAPIPE:
        mp_fd = mp.solutions.face_detection
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        with mp_fd.FaceDetection(model_selection=1, min_detection_confidence=min_confidence) as fd:
            results = fd.process(image_rgb)
        detections: List[Tuple[Tuple[int, int, int, int], float]] = []
        if results.detections:
            for det in results.detections:
                rel = det.location_data.relative_bounding_box
                x1 = int(rel.xmin * w)
                y1 = int(rel.ymin * h)
                x2 = int((rel.xmin + rel.width) * w)
                y2 = int((rel.ymin + rel.height) * h)
                x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, w, h)
                score = det.score[0] if det.score else 0.0
                detections.append(((x1, y1, x2, y2), float(score)))
        return detections

    # Fallback: Haar cascade (no confidence scores)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    dets: List[Tuple[Tuple[int, int, int, int], float]] = []
    for (x, y, bw, bh) in rects:
        x1, y1, x2, y2 = _clip_box(int(x), int(y), int(x + bw), int(y + bh), w, h)
        dets.append(((x1, y1, x2, y2), 1.0))
    return dets


def crop_regions(image_bgr: np.ndarray, boxes: List[Tuple[int, int, int, int]], margin_percent: int = 10) -> List[np.ndarray]:
    h, w = image_bgr.shape[:2]
    margin_percent = max(0, margin_percent)
    crops: List[np.ndarray] = []
    for (x1, y1, x2, y2) in boxes:
        bw, bh = x2 - x1, y2 - y1
        mx = int(bw * margin_percent / 100)
        my = int(bh * margin_percent / 100)
        cx1, cy1, cx2, cy2 = _clip_box(x1 - mx, y1 - my, x2 + mx, y2 + my, w, h)
        crop = image_bgr[cy1:cy2, cx1:cx2]
        if crop.size > 0:
            crops.append(crop)
    return crops
