import torch
import cv2
import numpy as np
from ultralytics import YOLO
from torchvision import transforms, models
from PIL import Image
import os

# ==============================
# CONFIGURATION
# ==============================
YOLO_WEIGHTS = "yolo_best.pt"        # path to YOLO weights
SWIN_WEIGHTS = "best_model.pth"     # path to Swin model
IMAGE_PATH   = "slide.jpg"          # input image
OUTPUT_PATH  = "output_result.jpg"

NUM_CLASSES = 6

CLASS_NAMES = [
    "Class_1",
    "Class_2",
    "Class_3",
    "Class_4",
    "Class_5",
    "Class_6"
]

CONF_THRES = 0.25

# ==============================
# DEVICE
# ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ==============================
# LOAD YOLO MODEL
# ==============================
yolo_model = YOLO(YOLO_WEIGHTS)

# ==============================
# LOAD SWIN TRANSFORMER
# ==============================
swin_model = models.swin_t(weights=None)
swin_model.head = torch.nn.Linear(
    swin_model.head.in_features, NUM_CLASSES
)

state_dict = torch.load(SWIN_WEIGHTS, map_location=DEVICE)
swin_model.load_state_dict(state_dict)

swin_model.to(DEVICE)
swin_model.eval()

# ==============================
# PREPROCESSING
# ==============================
swin_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ==============================
# LOAD IMAGE
# ==============================
image_bgr = cv2.imread(IMAGE_PATH)
if image_bgr is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# ==============================
# YOLO DETECTION
# ==============================
results = yolo_model(image_rgb, conf=CONF_THRES)

# ==============================
# DETECT → CROP → CLASSIFY
# ==============================
predictions = []

for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        crop = image_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_pil = Image.fromarray(crop)

        input_tensor = swin_transform(crop_pil)
        input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = swin_model(input_tensor)
            pred_id = torch.argmax(output, dim=1).item()

        predictions.append((x1, y1, x2, y2, pred_id))

# ==============================
# DRAW RESULTS
# ==============================
for (x1, y1, x2, y2, cls_id) in predictions:
    label = CLASS_NAMES[cls_id]

    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        image_bgr,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

# ==============================
# SAVE OUTPUT
# ==============================
cv2.imwrite(OUTPUT_PATH, image_bgr)
print(f"Output saved to: {OUTPUT_PATH}")
