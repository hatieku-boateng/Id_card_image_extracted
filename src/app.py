import io
import os
import sys
import zipfile
from dataclasses import dataclass
from typing import List, Tuple

# Allow running with `streamlit run src/app.py` by ensuring `src` is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

"""Optional runtime installer
If deploying to Streamlit Community Cloud, requirements.txt is sufficient and preferred.
This block allows installing missing packages at runtime when the env var
`ALLOW_RUNTIME_INSTALL` is set to 1/true. After installing, the app reruns once.
"""
try:
    from utils.bootstrap import ensure_packages  # type: ignore
except Exception:
    # utils may not exist; skip runtime install support
    ensure_packages = None  # type: ignore

installed_runtime = False
if ensure_packages is not None:
    try:
        installed_runtime = ensure_packages(allow_runtime=True)
    except Exception:
        installed_runtime = False

import streamlit as st
if installed_runtime:
    st.info("Installed missing packages. Rerunning once…")
    st.rerun()

import cv2
import numpy as np
from PIL import Image

from face_extractor.detector import detect_faces, crop_regions


@dataclass
class FaceResult:
    bbox: Tuple[int, int, int, int]
    score: float


def load_image_to_bgr(uploaded_file) -> np.ndarray:
    data = uploaded_file.read()
    pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def draw_bboxes(image: np.ndarray, boxes: List[Tuple[int, int, int, int]], main_index: int) -> np.ndarray:
    vis = image.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        color = (0, 255, 0) if i == main_index else (255, 200, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = "main" if i == main_index else "face"
        cv2.putText(vis, label, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


def to_download_bytes(img: np.ndarray, ext: str = ".jpg") -> bytes:
    ok, buf = cv2.imencode(ext, img)
    return buf.tobytes() if ok else b""


def main():
    st.set_page_config(page_title="ID Portrait Extractor", page_icon="🪪", layout="centered")
    st.title("ID Portrait Extractor")
    st.caption("No-training approach using MediaPipe face detection to crop portrait(s) from ID images.")

    with st.sidebar:
        st.header("Options")
        conf = st.slider("Min confidence", 0.1, 0.99, 0.6, 0.01)
        margin = st.slider("Crop margin (%)", 0, 40, 10, 1)
        mode = st.radio("Return", ["Largest only", "All faces"], index=0)
        max_faces = st.number_input("Max faces (when 'All faces')", min_value=1, max_value=10, value=5, step=1)
        st.markdown("---")
        st.markdown("Tip: If background faces are detected, capture a tighter photo of the ID.")

    uploaded = st.file_uploader("Upload an ID card image", type=["jpg", "jpeg", "png", "webp"])
    if not uploaded:
        st.stop()

    image_bgr = load_image_to_bgr(uploaded)
    h, w = image_bgr.shape[:2]

    detections = detect_faces(image_bgr, min_confidence=conf)

    if len(detections) == 0:
        st.warning("No faces detected. Try lowering the confidence or using a clearer image.")
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Input", use_container_width=True)
        st.stop()

    # Sort by area (descending) and keep up to max_faces
    detections.sort(key=lambda d: (d[0][2] - d[0][0]) * (d[0][3] - d[0][1]), reverse=True)
    if mode == "Largest only":
        detections = detections[:1]
    else:
        detections = detections[:max_faces]

    boxes = [d[0] for d in detections]
    main_idx = 0  # after sorting, first is largest
    vis = draw_bboxes(image_bgr, boxes, main_idx)
    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Detections", use_container_width=True)

    crops = crop_regions(image_bgr, boxes, margin_percent=margin)

    if len(crops) == 0:
        st.error("Failed to crop faces.")
        st.stop()

    st.subheader("Cropped Portraits")
    cols = st.columns(min(3, len(crops)))
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, crop in enumerate(crops):
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            cols[i % len(cols)].image(rgb, caption=f"portrait_{i}")
            # Add to ZIP
            zf.writestr(f"portrait_{i}.jpg", to_download_bytes(crop))

    # Main portrait download
    main_bytes = to_download_bytes(crops[0])
    st.download_button("Download main portrait", data=main_bytes, file_name="portrait_main.jpg", mime="image/jpeg")

    # ZIP download
    zip_buffer.seek(0)
    st.download_button("Download all as ZIP", data=zip_buffer, file_name="portraits.zip", mime="application/zip")


if __name__ == "__main__":
    main()
