import os
import random
import shutil
import cv2
import numpy as np

# === Step 1: Define paths ===
origimages = "LPDataset/train/images"
origlabels = "LPDataset/train/labels"
darkimages = "LPDataset/train/images_dark"
darklabels = "LPDataset/train/labels_dark"
normalimages = "LPDataset/train/images_subset"
normallabels = "LPDataset/train/labels_subset"

os.makedirs(darkimages, exist_ok=True)
os.makedirs(darklabels, exist_ok=True)
os.makedirs(normalimages, exist_ok=True)
os.makedirs(normallabels, exist_ok=True)

# === Step 2: Randomly sample 2000 total, split in half ===
sampled = random.sample(os.listdir(origimages), 2000)
dark_sample = sampled[:1000]
normal_sample = sampled[1000:]

# === Step 3: Copy and darken half ===
for fname in dark_sample:
    shutil.copy(os.path.join(origimages, fname), os.path.join(darkimages, fname))
    label = fname.replace('.jpg', '.txt').replace('.png', '.txt')
    shutil.copy(os.path.join(origlabels, label), os.path.join(darklabels, label))

    # Darken the copied image
    path = os.path.join(darkimages, fname)
    img = cv2.imread(path)
    if img is not None:
        dark = (img * 0.4).clip(0, 255).astype("uint8")
        cv2.imwrite(path, dark)

# === Step 4: Copy the other half as-is (normal lighting) ===
for fname in normal_sample:
    shutil.copy(os.path.join(origimages, fname), os.path.join(normalimages, fname))
    label = fname.replace('.jpg', '.txt').replace('.png', '.txt')
    shutil.copy(os.path.join(origlabels, label), os.path.join(normallabels, label))
