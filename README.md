# Property_Damage_Segmentation_Yolo

Property_Damage_Segmentation_Yolo is a computer vision project that detects and segments property damages (cracks, corrosion-induced spalling, spalling, peeling) using YOLOv11 and YOLOv12 segmentation models. It delivers precise pixel-level damage outlines for accurate assessment.

---

## ✨ Features
- Automatic detection of multiple property damage types.
- Pixel-level segmentation for precise localization.
- Built on YOLO (You Only Look Once) deep learning framework.
- Supports YOLOv11 and YOLOv12 segmentation models.

---

✅ How TO Run This APP?
Build the base image first:

    ```
    docker build -t crack-seg-base -f Dockerfile.base .
    ```

Build and run services with Compose:

    ```
    docker compose up --build
    ```

Streamlit depends on FastAPI → Compose waits for FastAPI container to start before starting Streamlit.
Both services share the same dependencies installed in the base image, so you don’t reinstall twice.
