import os
import shutil
import pandas as pd

# Folder EXACTLY as seen in your test
IMAGE_FOLDER = "ISIC-images (1)"

CSV_PATH = "metadata.csv"

OUTPUT_DIR = "separated_images"
BENIGN_DIR = os.path.join(OUTPUT_DIR, "benign")
MALIGNANT_DIR = os.path.join(OUTPUT_DIR, "malignant")

os.makedirs(BENIGN_DIR, exist_ok=True)
os.makedirs(MALIGNANT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

def get_label(row):
    d1 = str(row["diagnosis_1"]).lower()
    if "malignant" in d1:
        return "malignant"
    if "benign" in d1:
        return "benign"
    return None

benign_moved = 0
malignant_moved = 0
missing = 0
skipped = 0

for _, row in df.iterrows():
    isic_id = row["isic_id"]

    label = get_label(row)
    if label is None:
        skipped += 1
        continue

    found = False
    for ext in [".jpg", ".JPG", ".jpeg", ".JPEG", ".png"]:
        src = os.path.join(IMAGE_FOLDER, isic_id + ext)
        if os.path.exists(src):
            if label == "benign":
                dst = os.path.join(BENIGN_DIR, isic_id + ext)
                benign_moved += 1
            else:
                dst = os.path.join(MALIGNANT_DIR, isic_id + ext)
                malignant_moved += 1

            shutil.copy2(src, dst)
            found = True
            break

    if not found:
        missing += 1

print("\n============ SUMMARY ============")
print("Benign images moved:    ", benign_moved)
print("Malignant images moved: ", malignant_moved)
print("Missing images:         ", missing)
print("Skipped rows:           ", skipped)
print("Output folders:")
print("  ", BENIGN_DIR)
print("  ", MALIGNANT_DIR)
print("=================================\n")
