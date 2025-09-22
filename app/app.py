import os
import streamlit as st
import requests
from PIL import Image

# ----------------------------
# API config
# ----------------------------
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict/")

# ----------------------------
# Enum-like model options
# ----------------------------
MODEL_CHOICES = {
    "YOLOv12 Segmentation": "yolov12-seg",
    "YOLOv11 Segmentation": "yolov11-seg",
    "YOLOv12 Batch 16 Segmentation": "yolov12-batch16-seg"
}

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("Model Selection")
model_choice_display = st.sidebar.selectbox("Choose a segmentation model:", list(MODEL_CHOICES.keys()))
model_choice = MODEL_CHOICES[model_choice_display]

st.title("🛠️ Damage Segmentation Demo")

# ----------------------------
# File Upload
# ----------------------------
uploaded_file = st.file_uploader("Upload an image for segmentation", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Segmentation"):
        # Send image + model choice to API
        response = requests.post(
            API_URL,
            params={"model_name": model_choice},
            files={"file": (uploaded_file.name, uploaded_file.getvalue())}
        )

        if response.status_code == 200:
            result = response.json()
            st.success(f"Segmentation completed using **{result.get('model_used', model_choice)}**")

            # ✅ Show metadata if available
            if "image_height" in result and "image_width" in result:
                st.write(f"**Image Size:** {result['image_width']} x {result['image_height']}")
            if "total_area" in result:
                st.write(f"**Total Area in pixel² is:** {result['total_area']:,} pixel²")
                st.write(f"**Total Area cm² is :** {round(0.000577 * result['total_area'], 3):,} cm²")
                st.write(f"**Total Area in mm²:** {round(0.0577 * result['total_area']):,} mm²")

            # ✅ Show per-class statistics if available
            if "class_statistics" in result and result["class_statistics"]:
                st.subheader("📊 Class-wise Statistics")
                for cls, stats in result["class_statistics"].items():
                    st.markdown(f"### {cls}")
                    st.write(f"- Detections: {stats['detections']}")
                    st.write(f"- Mask Area in pixel² is: {stats['total_mask_area']:,} pixel²")
                    st.write(f"- Mask Area in cm² is: {round(0.000577 * stats['total_mask_area'], 3):,} cm²")
                    st.write(f"- Mask Area in mm² is: {round(0.0577 * stats['total_mask_area'], 3):,} mm²")
                    st.write(f"- Relative Damage: {stats['relative_damage_percent']:.2f} %")

            # ✅ Show overall damage intensity
            if "overall_damage_percent" in result:
                st.subheader("🔥 Overall Damage Intensity")
                st.write(f"**{result['overall_damage_percent']:.2f} %**")

            # ✅ Show predicted labels (if still provided by API)
            if result.get("labels"):
                st.subheader("Predicted Classes")
                st.write(", ".join(result["labels"]))

            # ✅ Show segmented result image
            output_path = result.get("output_path")
            if output_path and os.path.exists(output_path):
                st.image(output_path, caption="Segmented Result", use_column_width=True)
            else:
                st.warning("⚠️ Output image not found on server. (Consider returning image bytes from API)")

        else:
            st.error(f"Error: {response.text}")
