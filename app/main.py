import os
import cv2
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from datetime import datetime
from enum import Enum

# ----------------------------
# Configuration
# ----------------------------
OUTPUT_DIR = "outputs"
CONF_THRESHOLD = 0.2
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Enum for model choices
# ----------------------------
class ModelChoice(str, Enum):
    yolov12_seg = "yolov12-seg"
    yolov11_seg = "yolov11-seg"
    yolov12_batch16_seg = "yolov12-batch16-seg"
    # unet_seg = "unet-seg"



MODEL_DIR = os.getenv("MODEL_DIR", "models")
model_map = {
    ModelChoice.yolov12_seg: os.path.join(MODEL_DIR, "yolov12_seg_best.pt"),
    ModelChoice.yolov11_seg: os.path.join(MODEL_DIR, "yolov11_seg_best.pt"),
    ModelChoice.yolov12_batch16_seg: os.path.join(MODEL_DIR, "best.pt"),
}


# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="Crack Detection API")

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    model_name: ModelChoice = Query(..., description="Choose a segmentation model")
):
    try:
        if model_name in [ModelChoice.yolov12_seg, ModelChoice.yolov11_seg, ModelChoice.yolov12_batch16_seg]:
            model = YOLO(model_map[model_name])
            # Save uploaded file temporarily
            input_path = os.path.join(OUTPUT_DIR, file.filename)
            with open(input_path, "wb") as f:
                f.write(await file.read())

            # Load original image to compute full area
            image = cv2.imread(input_path)
            height, width = image.shape[:2]
            total_area = height * width

            # Run inference
            results = model(input_path, conf=CONF_THRESHOLD)

            # Generate unique output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{model_name}_{timestamp}_{file.filename}"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            # Store per-class statistics
            class_stats = {}
            total_mask_area = 0

            for r in results:
                # Save annotated image
                annotated = r.plot()
                cv2.imwrite(output_path, annotated)

                if r.masks is not None and r.boxes is not None:
                    masks = r.masks.data.cpu().numpy()   # [num_masks, H, W]
                    classes = r.boxes.cls.cpu().numpy()  # class IDs

                    for mask, cls_id in zip(masks, classes):
                        class_name = r.names[int(cls_id)]
                        mask_area = int(mask.sum())
                        total_mask_area += mask_area

                        if class_name not in class_stats:
                            class_stats[class_name] = {
                                "detections": 0,
                                "total_mask_area": 0
                            }
                        class_stats[class_name]["detections"] += 1
                        class_stats[class_name]["total_mask_area"] += mask_area

            # Compute relative intensities (in %)
            for cls_name, stats in class_stats.items():
                stats["relative_damage_percent"] = (
                    (stats["total_mask_area"] / total_area) * 100 if total_area > 0 else 0
                )

            overall_damage_percent = (total_mask_area / total_area) * 100 if total_area > 0 else 0

            return JSONResponse({
                "status": "success",
                "input_file": file.filename,
                "output_file": output_filename,
                "output_path": output_path,
                "model_used": model_name,
                "image_height": height,
                "image_width": width,
                "total_area": total_area,
                "class_statistics": class_stats,
                "overall_damage_percent": overall_damage_percent
            })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
