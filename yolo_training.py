import os
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import time
import yaml

# --------------------------
# CONFIGURATION
# --------------------------
MODEL = r"C:\Users\n2007\Desktop\malaria_detection\yolov8n.pt" 
DATA_YAML = r"C:\Users\n2007\Desktop\malaria_detection\dataset_yolo\data.yaml"
SAVE_DIR = r"C:\Users\n2007\Desktop\malaria_detection\runs\yolo_cells"
EPOCHS = 100
BATCH_SIZE = 8
DEVICE = "cpu"
EXP_NAME = "exp_cells"

# --------------------------
# TRAIN YOLO
# --------------------------
start_time = time.time()
model = YOLO(MODEL)

results = model.train(
    data=DATA_YAML,
    imgsz=640,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    patience=10,  # early stopping
    device=DEVICE,
    project=SAVE_DIR,
    name=EXP_NAME,
    verbose=True
)

# --------------------------
# LOGGING: PER EPOCH
# --------------------------
results_csv = os.path.join(SAVE_DIR, EXP_NAME, "results.csv")
log_txt = os.path.join(SAVE_DIR, EXP_NAME, "epoch_log.txt")

if os.path.exists(results_csv):
    df = pd.read_csv(results_csv)

    with open(log_txt, "w") as f:
        f.write("Epoch\tBoxLoss\tClsLoss\tPrecision\tRecall\tmAP50\tmAP50-95\n")
        for i, row in df.iterrows():
            f.write(f"{int(row['epoch'])}\t"
                    f"{row['train/box_loss']:.4f}\t"
                    f"{row['train/cls_loss']:.4f}\t"
                    f"{row['metrics/precision(B)']:.4f}\t"
                    f"{row['metrics/recall(B)']:.4f}\t"
                    f"{row['metrics/mAP50(B)']:.4f}\t"
                    f"{row['metrics/mAP50-95(B)']:.4f}\n")

    print(f"✅ Epoch-wise metrics logged to: {log_txt}")
else:
    print("⚠️ results.csv not found. Logging skipped.")

# --------------------------
# PERFORMANCE PLOTS
# --------------------------
if os.path.exists(results_csv):
    df = pd.read_csv(results_csv)

    # Overall mAP, precision, recall
    plt.figure(figsize=(8, 6))
    plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@0.5-0.95")
    plt.plot(df["epoch"], df["metrics/precision(B)"], label="Precision")
    plt.plot(df["epoch"], df["metrics/recall(B)"], label="Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("YOLOv8 Overall Detection Performance")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, EXP_NAME, "overall_metrics.png"))
    plt.show()

# --------------------------
# CLASS-WISE PERFORMANCE
# --------------------------
# Load class names from data.yaml
with open(DATA_YAML, 'r') as f:
    data_yaml = yaml.safe_load(f)
CLASS_NAMES = data_yaml['names']

# Evaluate best model on validation set to get class-wise metrics
print("\n🔍 Evaluating best model for class-wise metrics...")
metrics = model.val(data=DATA_YAML, device=DEVICE, split="val")
class_metrics = metrics.results_dict["metrics/per_class"]

if class_metrics is not None:
    plt.figure(figsize=(10, 6))
    for idx, cname in enumerate(CLASS_NAMES):
        mAP50_95 = class_metrics["mAP50-95"][idx]
        precision = class_metrics["precision"][idx]
        recall = class_metrics["recall"][idx]

        plt.bar(idx - 0.2, mAP50_95, width=0.2, label=f"{cname} mAP")
        plt.bar(idx, precision, width=0.2, label=f"{cname} Prec.")
        plt.bar(idx + 0.2, recall, width=0.2, label=f"{cname} Rec.")

    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45)
    plt.ylabel("Score")
    plt.title("Class-wise YOLOv8 Performance on Validation Set")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, EXP_NAME, "classwise_metrics.png"))
    plt.show()

end_time = time.time()
print(f"\n✅ Training completed in {(end_time - start_time)/60:.2f} minutes.")
print(f"Best weights saved at: {os.path.join(SAVE_DIR, EXP_NAME, 'weights', 'best.pt')}")
