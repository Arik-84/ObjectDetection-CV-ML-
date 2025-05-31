import shutil
import os

# Paths
image_dirs = ["images_dark", "images_subset"]
label_dirs = ["labels_dark", "labels_subset"]
base_path = "."

# Combined output
combined_images = os.path.join(base_path, "train_combined/images")
combined_labels = os.path.join(base_path, "train_combined/labels")

# Create output dirs
os.makedirs(combined_images, exist_ok=True)
os.makedirs(combined_labels, exist_ok=True)

# Copy images
for img_dir in image_dirs:
    img_path = os.path.join(base_path, img_dir)
    for fname in os.listdir(img_path):
        shutil.copy(os.path.join(img_path, fname), os.path.join(combined_images, fname))

# Copy labels
for lbl_dir in label_dirs:
    lbl_path = os.path.join(base_path, lbl_dir)
    for fname in os.listdir(lbl_path):
        shutil.copy(os.path.join(lbl_path, fname), os.path.join(combined_labels, fname))
