import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO
import gradio as gr

# --- Global Configurations ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

YOLO_MODEL_PATH = "app/Yolo/yolov11.pt"

# --- Model Loading ---
def load_models():
    """Loads the pre-trained YOLO and MiDaS models."""
    print("Loading YOLOv11 model...")
    yolo_model = YOLO(YOLO_MODEL_PATH)

    print("Loading MiDaS model...")
    midas_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
    midas_model.to(DEVICE)
    midas_model.eval()

    return yolo_model, midas_model

try:
    yolo_model, midas_model = load_models()
except Exception as e:
    print(f"Error loading models: {e}")
    yolo_model, midas_model = None, None

# --- Main Processing Pipeline ---------------
def process_image_pipeline(input_image_path):
    if not input_image_path:
        return None, None, None, None, "Please upload an image to process."

    # Original Image--------
    try:
        image = cv2.imread(input_image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_img = image_rgb.copy()
    except Exception as e:
        return None, None, None, None, f"Error loading image: {e}"

    # YOLO Object Detection---------
    results = yolo_model.predict(image_rgb, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    # Filter for vehicles: car (2), bus (5), truck (7)
    vehicle_boxes = [box for box, cls in zip(boxes, classes) if int(cls) in [2, 5, 7]]

    yolo_img = image_rgb.copy()
    for box in vehicle_boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(yolo_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(yolo_img, "Vehicle", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)

    # MiDaS Depth Estimation
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    input_tensor = transform(image_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prediction = midas_model(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Invert and normalize depth
    normalized_depth = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    depth_map = 1 - normalized_depth

    # Depth visualization
    depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_vis = np.uint8(depth_vis)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
    depth_colored_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

    # Distance Calculation & Visualization-------------
    scaling_factor = 17  # calibrate for real-world use
    vis_img = image_rgb.copy()
    decision_text = " Driving Decisions:\n\n"

    if not vehicle_boxes:
        decision_text = " Safe — No vehicles detected."
    else:
        for i, box in enumerate(vehicle_boxes):
            x1, y1, x2, y2 = box.astype(int)

            # Ensure bounding box coordinates are within image bounds
            x1 = max(x1, 0); y1 = max(y1, 0)
            x2 = min(x2, depth_map.shape[1]); y2 = min(y2, depth_map.shape[0])

            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue

            # Take bottom N pixels for smoothing (reduce noise)
            N = min(5, y2 - y1)
            bottom_strip = depth_map[y2 - N:y2, x1:x2]
            if bottom_strip.size == 0:
                continue

            min_depth_value = np.max(bottom_strip)  # max since depth is inverted
            distance_in_meters = scaling_factor * min_depth_value

            # Draw bounding box and distance
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_img, f"{distance_in_meters:.2f} m",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 2)

            # Decision thresholds
            decision_text += f"Vehicle {i+1}:\n"
            decision_text += f" Distance = {distance_in_meters:.2f} meters\n"
            if distance_in_meters < 10.0:
                decision_text += " 'VERY CLOSE' — Apply Brake\n"
            elif distance_in_meters < 15.0:
                decision_text += " 'Caution' — Maintain Distance\n"
            else:
                decision_text += " 'Safe' — Continue Driving\n"
            decision_text += "-" * 45 + "\n"

    return original_img, yolo_img, depth_colored_rgb, vis_img, decision_text

# --- Gradio Dashboard -----------------
def run_gradio_app():
    iface = gr.Interface(
        fn=process_image_pipeline,
        inputs=gr.Image(type="filepath", label="Upload an Image"),
        outputs=[
            gr.Image(label="1. Original Image"),
            gr.Image(label="2. YOLO Bounding Boxes"),
            gr.Image(label="3. MiDaS Depth Heatmap"),
            gr.Image(label="4. Final Result with Distance"),
            gr.Textbox(label="Decision Log")
        ],
        title="Autonomous Vehicle Perception Pipeline Dashboard",
        description="Visualizes object detection, depth estimation, and perpendicular distance calculation for vehicles."
    )
    iface.launch(server_name="127.0.0.1")

# --- Main Execution ---
if __name__ == "__main__":
    run_gradio_app()
