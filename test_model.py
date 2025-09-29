import torch
import pathlib
import json
import os

# --- Fix for PosixPath when loading Linux-trained weights on Windows ---
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load model
model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path='C:/Users/Manthan/OneDrive/Desktop/microyolo/yolov5/best.pt',
    force_reload=True
)

# Restore PosixPath
pathlib.PosixPath = temp

# --- Inference settings ---
img_path = 'C:/Users/Manthan/OneDrive/Desktop/microyolo/yolov5/test_image2.jpg'
img_size = 1920
conf_threshold = 0.2  # filter threshold

# Run inference
results = model(img_path, size=img_size, augment=True)

# Get predictions
predictions = results.xyxy[0]
filtered = predictions[predictions[:, 4] > conf_threshold]  # conf > threshold

# Save annotated image
save_dir = 'C:/Users/Manthan/OneDrive/Desktop/microyolo/yolov5'
results.save(save_dir=save_dir)
annotated_img_path = os.path.join(save_dir, os.path.basename(img_path))

# --- Prepare JSON report ---
json_data = {
    "image_name": os.path.basename(img_path),
    "total_particles": len(filtered),
    "particles": []
}

# Get class names
class_names = model.names

# Populate JSON
for *box, conf, cls in filtered.tolist():
    x1, y1, x2, y2 = map(float, box)
    json_data["particles"].append({
        "type": class_names[int(cls)],
        "confidence": float(conf),
        "bbox": [x1, y1, x2, y2]
    })

# Save JSON file
json_path = os.path.join(save_dir, os.path.splitext(os.path.basename(img_path))[0] + ".json")
with open(json_path, "w") as f:
    json.dump(json_data, f, indent=4)

# Print info
print(f"✅ Detection complete. Annotated image saved in: {save_dir}")
print(f"✅ JSON report saved as: {json_path}")
print(f"Filtered detections (conf > {conf_threshold}):")
print(filtered)
