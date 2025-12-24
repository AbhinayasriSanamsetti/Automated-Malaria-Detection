import os
import json
import random
import shutil
from tqdm import tqdm
import yaml

# --------------------------
# CONFIGURATION
# --------------------------
BASE_DIR = "malaria"
IMAGES_DIR = os.path.join(BASE_DIR,"C:/Users/n2007/Desktop/malaria_detection/malaria/malaria/images")
TRAIN_JSON = os.path.join(BASE_DIR, "C:/Users/n2007/Desktop/malaria_detection/malaria/malaria/training.json")
TEST_JSON = os.path.join(BASE_DIR, "C:/Users/n2007/Desktop/malaria_detection/malaria/malaria/test.json")
OUTPUT_DIR = "dataset_yolo"

SPLIT_RATIO = 0.85  # % of training images used for training (rest for validation)
CLASSES = [
    "red blood cell",
    "trophozoite",
    "ring",
    "schizont",
    "gametocyte",
    "leukocyte"
]

# --------------------------
# HELPERS
# --------------------------
def make_dirs():
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

def get_class_id(category):
    if category in CLASSES:
        return CLASSES.index(category)
    else:
        return len(CLASSES)  # unknown class

# Convert bounding box to YOLO format
def convert_to_yolo(size, box):
    img_h, img_w = size
    x_min, y_min, x_max, y_max = box

    # Normalize to [0,1]
    x_center = ((x_min + x_max) / 2.0) / img_w
    y_center = ((y_min + y_max) / 2.0) / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    return x_center, y_center, width, height

# --------------------------
# PROCESS FUNCTION
# --------------------------
def process_json(json_path, split):
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Processing {len(data)} images for {split} set...")
    for item in tqdm(data):
        img_rel_path = item["image"]["pathname"].lstrip("/")  # remove leading /
        img_name = os.path.basename(img_rel_path)
        img_src = os.path.join(BASE_DIR, img_rel_path)
        if not os.path.exists(img_src):
            img_src = os.path.join(IMAGES_DIR, img_name)  # fallback

        img_h = item["image"]["shape"]["r"]
        img_w = item["image"]["shape"]["c"]

        label_name = os.path.splitext(img_name)[0] + ".txt"

        dst_img = os.path.join(OUTPUT_DIR, split, "images", img_name)
        dst_lbl = os.path.join(OUTPUT_DIR, split, "labels", label_name)

        # copy image
        if os.path.exists(img_src):
            shutil.copy2(img_src, dst_img)
        else:
            continue

        # write labels
        with open(dst_lbl, "w") as f:
            for obj in item["objects"]:
                category = obj["category"].strip().lower()
                if category not in CLASSES:
                    continue
                class_id = get_class_id(category)

                bb = obj["bounding_box"]
                x_min = bb["minimum"]["c"]
                y_min = bb["minimum"]["r"]
                x_max = bb["maximum"]["c"]
                y_max = bb["maximum"]["r"]

                x_center, y_center, w, h = convert_to_yolo((img_h, img_w), (x_min, y_min, x_max, y_max))
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":
    make_dirs()

    # 1️⃣ Load and split training data into train and val
    with open(TRAIN_JSON, 'r') as f:
        all_train_data = json.load(f)

    random.shuffle(all_train_data)
    split_idx = int(len(all_train_data) * SPLIT_RATIO)
    train_data = all_train_data[:split_idx]
    val_data = all_train_data[split_idx:]

    # Save temporary split JSONs (optional)
    with open("train_split.json", "w") as f: json.dump(train_data, f)
    with open("val_split.json", "w") as f: json.dump(val_data, f)

    # 2️⃣ Process and export YOLO files
    process_json("train_split.json", "train")
    process_json("val_split.json", "val")
    process_json(TEST_JSON, "test")

    # 3️⃣ Create YOLO data.yaml
    data_yaml = {
        "train": os.path.abspath(os.path.join(OUTPUT_DIR, "train", "images")),
        "val": os.path.abspath(os.path.join(OUTPUT_DIR, "val", "images")),
        "test": os.path.abspath(os.path.join(OUTPUT_DIR, "test", "images")),
        "nc": len(CLASSES),
        "names": CLASSES
    }

    with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w") as f:
        yaml.dump(data_yaml, f)

    print("\n✅ YOLO dataset prepared successfully!")
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: from test.json")
    print(f"data.yaml saved at: {os.path.join(OUTPUT_DIR, 'data.yaml')}")
