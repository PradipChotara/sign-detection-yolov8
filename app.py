import time
from collections import deque
from pathlib import Path
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
import av
import cv2

# --- Custom CSS for scrollable sidebar and styled scrollbar ---
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            overflow-y: auto !important;
        }
        ::-webkit-scrollbar {
            width: 10px;
        }
        ::-webkit-scrollbar-thumb {
            background: #888;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="YOLOv8n Real-Time Webcam Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar content: Show images with their names ---
with st.sidebar:
    st.title("Photo Gallery")
    photos_dir = Path("assets/photos")
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    if photos_dir.exists() and photos_dir.is_dir():
        photo_files = [f for f in photos_dir.iterdir() if f.suffix.lower() in image_extensions]
        if not photo_files:
            st.write("No photos found in assets/photos.")
        else:
            for photo in photo_files:
                st.image(str(photo), width=180)
                st.caption(photo.stem)
    else:
        st.write("No photos found in assets/photos.")

# --- Load YOLOv8 model ---
MODEL_PATH = Path(__file__).parent / "assets" / "model" / "best.pt"
model = YOLO(str(MODEL_PATH))

# --- FPS calculation helper ---
frame_times = deque(maxlen=30)  # Smoother FPS

def process_frame(frame):
    img = frame.to_ndarray(format="bgr24")
    current_time = time.time()
    frame_times.append(current_time)
    if len(frame_times) > 1:
        fps = len(frame_times) / (frame_times[-1] - frame_times[0])
    else:
        fps = 0.0

    # YOLO inference
    results = model(img)
    annotated_frame = results[0].plot()

    # Draw FPS on frame
    fps_text = f"FPS: {fps:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 255, 0)
    thickness = 2
    text_size, _ = cv2.getTextSize(fps_text, font, font_scale, thickness)
    text_x = annotated_frame.shape[1] - text_size[0] - 10
    text_y = 30
    cv2.putText(annotated_frame, fps_text, (text_x, text_y), font, font_scale, color, thickness)

    # --- Resize the frame here ---
    target_width, target_height = 320, 240  # Change to your preferred size
    resized_frame = cv2.resize(annotated_frame, (target_width, target_height))

    return av.VideoFrame.from_ndarray(resized_frame, format="bgr24")

# --- Main page content ---
st.title("Fine-tuned YOLOv8n Sign Detection")

col1, col2, col3 = st.columns([1,2,1])
with col2:
    webrtc_streamer(
        key="yolov8",
        video_frame_callback=process_frame,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        mode=WebRtcMode.SENDRECV,
    )

# --- Automated Labeling with MediaPipe ---
st.markdown("""
## Automated Labeling with MediaPipe

- Used **MediaPipe Hands** to detect hand landmarks in each image.
- Calculated the smallest enclosing bounding box from landmark coordinates.
- Normalized bounding box center, width, and height to YOLO format.
- Class labels were automatically extracted from filenames and mapped to numeric IDs.
- Generated YOLO-compatible `.txt` annotation files for each image.
- Entire process was fully automated—no manual labeling required.
- Ensured consistent, reproducible, and scalable dataset annotation.
""")

# --- Data Augmentation and Model Training Details Side by Side ---
colA, colB = st.columns(2)

with colA:
    st.markdown("""
    ### Augmentation Techniques Used

    | Augmentation Technique   | Description                                   |
    |-------------------------|-----------------------------------------------|
    | Rotation                | Random rotation within ±30°                   |
    | Scaling                 | Random scaling up to ±20%                     |
    | Shearing                | Affine shear up to 20°                        |
    | Brightness & Contrast   | Random brightness and contrast adjustment     |
    | Gaussian Blurring       | Blur with small kernel                        |
    | Sharpening              | Edge and detail enhancement                   |
    """)

with colB:
    st.markdown("""
    ### Model Training Summary

    | Parameter   | Value                               |
    |:------------|:------------------------------------|
    | Base Model  | yolov8n.pt                          |
    | Epochs      | 20                                  |
    | Img Size    | 640                                 |
    | Batch       | 16                                  |
    | Optimizer   | AdamW                               |
    | AMP         | Enabled                             |
    | Data        | 756 train, 216 val, 108 test images |
    | Classes     | 1-9                                 |
    """)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ## YOLOv8n Training Results Per Epoch

    | Epoch | Precision | Recall | mAP50 | mAP50-95 |
    |-------|-----------|--------|-------|----------|
    | 1     | 0.734     | 0.0751 | 0.373 | 0.303    |
    | 2     | 0.796     | 0.668  | 0.796 | 0.651    |
    | 3     | 0.738     | 0.830  | 0.869 | 0.724    |
    | 4     | 0.906     | 0.821  | 0.918 | 0.771    |
    | 5     | 0.950     | 0.908  | 0.961 | 0.812    |
    | 6     | 0.910     | 0.947  | 0.961 | 0.849    |
    | 7     | 0.932     | 0.957  | 0.981 | 0.853    |
    | 8     | 0.938     | 0.954  | 0.977 | 0.859    |
    | 9     | 0.958     | 0.914  | 0.969 | 0.876    |
    | 10    | 0.977     | 0.969  | 0.977 | 0.878    |
    | 11    | 0.977     | 0.950  | 0.973 | 0.874    |
    | 12    | 0.968     | 0.970  | 0.975 | 0.869    |
    | 13    | 0.968     | 0.949  | 0.973 | 0.895    |
    | 14    | 0.972     | 0.956  | 0.975 | 0.896    |
    | 15    | 0.979     | 0.964  | 0.967 | 0.903    |
    | 16    | 0.981     | 0.962  | 0.976 | 0.911    |
    | 17    | 0.972     | 0.958  | 0.978 | 0.913    |
    | 18    | 0.975     | 0.956  | 0.978 | 0.928    |
    | 19    | 0.984     | 0.953  | 0.980 | 0.924    |
    | 20    | 0.981     | 0.961  | 0.981 | 0.926    |
    """)

with col2:
    st.markdown("""
    ## Final Model Validation Results (Per Class)

    | Class | Instances | Precision | Recall | mAP50 | mAP50-95 |
    |-------|-----------|-----------|--------|-------|----------|
    | 1     | 23        | 0.935     | 0.913  | 0.934 | 0.861    |
    | 2     | 21        | 1.000     | 0.873  | 0.912 | 0.882    |
    | 3     | 20        | 0.937     | 1.000  | 0.995 | 0.961    |
    | 4     | 24        | 0.986     | 0.917  | 0.992 | 0.922    |
    | 5     | 24        | 0.976     | 1.000  | 0.995 | 0.995    |
    | 6     | 24        | 0.983     | 1.000  | 0.995 | 0.916    |
    | 7     | 21        | 0.984     | 0.905  | 0.989 | 0.862    |
    | 8     | 23        | 0.993     | 1.000  | 0.995 | 0.976    |
    | 9     | 24        | 0.981     | 1.000  | 0.995 | 0.967    |
    | **All** | 204     | 0.975     | 0.956  | 0.978 | 0.927    |
    """)

st.markdown("## YOLOv8n Training Metrics per Epoch")
st.image("assets/training_metrics_plot.jpg", caption="YOLOv8n training metrics (Precision, Recall, mAP50, mAP50-95) per epoch.")
