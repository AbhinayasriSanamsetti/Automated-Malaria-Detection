import os
import cv2
import random
import shutil
import numpy as np
import yaml
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image, ImageEnhance
from collections import Counter
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# CONFIGURATION
# ----------------------------
MODEL_PATH = r"C:\Users\n2007\Desktop\malaria_detection\runs\yolo_cells\exp_cells\weights\best.pt"
DATA_YAML = r"C:\Users\n2007\Desktop\malaria_detection\dataset_yolo\data.yaml"
SAVE_ROOT = r"C:\Users\n2007\Desktop\malaria_detection\final_swin_dataset"

IMG_SIZE = 224
AUGMENT_PROB = 0.3
TARGET_COUNT = 25000  # per class for balancing
SPLIT_RATIOS = [0.7, 0.15, 0.15]  # train, val, test
MASK_BACKGROUND = "black"  # or "white"

# ----------------------------
# UTILITIES
# ----------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def random_augment(image):
    """Apply random augmentation."""
    aug_type = random.choice(["rotate", "flip", "brightness", "contrast", "blur", "none"])
    if aug_type == "rotate":
        return image.rotate(random.choice([90, 180, 270]))
    elif aug_type == "flip":
        return image.transpose(random.choice([Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]))
    elif aug_type == "brightness":
        return ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))
    elif aug_type == "contrast":
        return ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))
    elif aug_type == "blur":
        img_cv = cv2.GaussianBlur(np.array(image), (3, 3), 0)
        return Image.fromarray(img_cv)
    else:
        return image

def preprocess_image(image):
    """Resize and normalize image."""
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image).astype(np.float32)
    arr = np.clip(arr, 0, 255) / 255.0
    return Image.fromarray((arr * 255).astype(np.uint8))

def plot_class_distribution(counter_before, counter_after, save_dir):
    """Visualize class distribution before and after balancing."""
    plt.figure(figsize=(10, 6))
    labels = sorted(set(counter_before.keys()) | set(counter_after.keys()))
    before = [counter_before.get(l, 0) for l in labels]
    after = [counter_after.get(l, 0) for l in labels]

    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width/2, before, width, label='Before')
    plt.bar(x + width/2, after, width, label='After')

    plt.xlabel("Class")
    plt.ylabel("Image Count")
    plt.title("Class Distribution Before vs After Balancing")
    plt.xticks(x, labels, rotation=30)
    plt.legend()
    plt.tight_layout()

    graph_path = os.path.join(save_dir, "class_distribution.png")
    plt.savefig(graph_path)
    plt.close()
    print(f"\n📊 Saved class distribution graph at: {graph_path}")

# ----------------------------
# STEP 1: YOLO DETECTION + CIRCULAR CROPPING
# ----------------------------
def crop_cells_yolo(model, data_yaml, save_dir):
    with open(data_yaml, "r") as f:
        yaml_data = yaml.safe_load(f)
    class_names = yaml_data["names"]

    img_dirs = [yaml_data[k] for k in ["train", "val", "test"] if k in yaml_data]
    img_files = []
    for d in img_dirs:
        if os.path.exists(d):
            img_files.extend([os.path.join(d, f) for f in os.listdir(d)
                              if f.lower().endswith((".jpg", ".png", ".jpeg"))])

    print(f"\n🚀 Step 1: YOLO detection and circular cropping (All slides)")
    print(f"🔍 Found {len(img_files)} total slides/images in dataset.")

    all_data = []
    for img_path in tqdm(img_files, desc="Running YOLO on all slides"):
        img = cv2.imread(img_path)
        if img is None:
            continue

        results = model.predict(img, verbose=False)
        boxes = results[0].boxes

        for i, box in enumerate(boxes):
            cls_id = int(box.cls)
            label = class_names[cls_id]
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Circular mask creation
            h, w = crop.shape[:2]
            center = (w // 2, h // 2)
            radius = min(center)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)

            masked = cv2.bitwise_and(crop, crop, mask=mask)

            # Fill background color outside circle
            bg_color = (0, 0, 0) if MASK_BACKGROUND == "black" else (255, 255, 255)
            bg = np.full_like(crop, bg_color)
            circular_crop = np.where(mask[:, :, None] == 255, masked, bg)

            label_dir = os.path.join(save_dir, label)
            ensure_dir(label_dir)
            base = os.path.splitext(os.path.basename(img_path))[0]
            crop_name = f"{base}_{i}_{label}.jpg"
            crop_path = os.path.join(label_dir, crop_name)
            cv2.imwrite(crop_path, circular_crop)

            all_data.append({
                "path": crop_path,
                "label": label,
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })

    return all_data, class_names

# ----------------------------
# STEP 2: PREPROCESS + AUGMENT
# ----------------------------
def preprocess_and_augment(data):
    processed = []
    for item in tqdm(data, desc="Preprocessing + Augmentation"):
        img = Image.open(item["path"]).convert("RGB")
        img = preprocess_image(img)
        img.save(item["path"])
        processed.append(item)

        if random.random() < AUGMENT_PROB:
            aug_img = random_augment(img)
            aug_path = item["path"].replace(".jpg", "_aug.jpg")
            aug_img.save(aug_path)
            processed.append({
                "path": aug_path,
                "label": item["label"],
                "bbox": item["bbox"]
            })
    return processed

# ----------------------------
# STEP 3: BALANCING
# ----------------------------
def balance_classes(data, save_root):
    labels = [x["label"] for x in data]
    counts_before = Counter(labels)
    print("\n📋 Class distribution before balancing:")
    for k, v in counts_before.items():
        print(f"{k}: {v}")

    balanced = []
    for label, count in counts_before.items():
        samples = [x for x in data if x["label"] == label]

        if count > TARGET_COUNT:
            balanced.extend(random.sample(samples, TARGET_COUNT))
        else:
            diff = TARGET_COUNT - count
            balanced.extend(samples)
            for _ in range(diff):
                sample = random.choice(samples)
                img = Image.open(sample["path"]).convert("RGB")
                aug_img = random_augment(img)
                aug_path = sample["path"].replace(".jpg", f"_up{_}.jpg")
                aug_img.save(aug_path)
                balanced.append({
                    "path": aug_path,
                    "label": label,
                    "bbox": sample["bbox"]
                })

    counts_after = Counter([x["label"] for x in balanced])
    print("\n✅ Balanced class distribution:")
    for k, v in counts_after.items():
        print(f"{k}: {v}")

    plot_class_distribution(counts_before, counts_after, save_root)
    return balanced

# ----------------------------
# STEP 4: SPLIT + SAVE + LABEL FILE
# ----------------------------
def split_and_save(data, save_root):
    df = pd.DataFrame(data)
    train_df, temp_df = train_test_split(df, stratify=df["label"], test_size=(1 - SPLIT_RATIOS[0]), random_state=42)
    val_df, test_df = train_test_split(temp_df, stratify=temp_df["label"], test_size=SPLIT_RATIOS[2]/sum(SPLIT_RATIOS[1:]), random_state=42)
    splits = {"train": train_df, "val": val_df, "test": test_df}

    for split, split_df in splits.items():
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Saving {split}"):
            dst_dir = os.path.join(save_root, split, row["label"])
            ensure_dir(dst_dir)
            shutil.copy2(row["path"], dst_dir)

    label_txt = os.path.join(save_root, "all_labels.txt")
    with open(label_txt, "w") as f:
        f.write("split,image_path,label,bbox[x1,y1,x2,y2]\n")
        for split, df_ in splits.items():
            for _, row in df_.iterrows():
                f.write(f"{split},{row['path']},{row['label']},{row['bbox']}\n")

    print("\n✅ Final dataset prepared.")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Labels file: {label_txt}")

# ----------------------------
# MAIN PIPELINE
# ----------------------------
if __name__ == "__main__":
    ensure_dir(SAVE_ROOT)
    model = YOLO(MODEL_PATH)

    yolo_data, class_names = crop_cells_yolo(model, DATA_YAML, os.path.join(SAVE_ROOT, "cropped_cells"))
    processed = preprocess_and_augment(yolo_data)
    balanced = balance_classes(processed, SAVE_ROOT)
    split_and_save(balanced, SAVE_ROOT)
    print("\n🎯 Done! Swin Transformer dataset ready at:", SAVE_ROOT)
