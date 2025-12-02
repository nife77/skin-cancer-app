import os
import shutil
import random

# --- CONFIG (matches YOUR folder structure) ---
DATA_DIR = "data"
SOURCE_BENIGN = os.path.join(DATA_DIR, "benign")
SOURCE_MALIGNANT = os.path.join(DATA_DIR, "malignant")

TRAIN_BENIGN = os.path.join(DATA_DIR, "train", "benign")
TRAIN_MALIGNANT = os.path.join(DATA_DIR, "train", "malignant")

VAL_BENIGN = os.path.join(DATA_DIR, "val", "benign")
VAL_MALIGNANT = os.path.join(DATA_DIR, "val", "malignant")

TRAIN_RATIO = 0.8
random.seed(42)
# ------------------------------------------------

# Create output folders if they don’t exist
for folder in [TRAIN_BENIGN, TRAIN_MALIGNANT, VAL_BENIGN, VAL_MALIGNANT]:
    os.makedirs(folder, exist_ok=True)


def split_and_move(src_folder, train_folder, val_folder):
    files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    random.shuffle(files)

    total = len(files)
    train_count = int(total * TRAIN_RATIO)

    train_files = files[:train_count]
    val_files = files[train_count:]

    print(f"\nProcessing {src_folder}...")
    print(f"Total: {total}, Train: {len(train_files)}, Val: {len(val_files)}")

    # Move train files
    for f in train_files:
        shutil.move(os.path.join(src_folder, f), os.path.join(train_folder, f))

    # Move val files
    for f in val_files:
        shutil.move(os.path.join(src_folder, f), os.path.join(val_folder, f))


# Split benign
split_and_move(SOURCE_BENIGN, TRAIN_BENIGN, VAL_BENIGN)

# Split malignant
split_and_move(SOURCE_MALIGNANT, TRAIN_MALIGNANT, VAL_MALIGNANT)

print("\nDone! Train/Val split complete.")
print("Train folders:", TRAIN_BENIGN, "and", TRAIN_MALIGNANT)
print("Val folders:", VAL_BENIGN, "and", VAL_MALIGNANT)
