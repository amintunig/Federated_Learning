import os
import shutil
import pandas as pd
from collections import defaultdict

# Paths
base_path = "D:/Ascl_Mimic_Data/SkinCancerMNIST"
image_folder_1 = os.path.join(base_path, "HAM10000_images_part_1")
image_folder_2 = os.path.join(base_path, "HAM10000_images_part_2")
metadata_path = os.path.join(base_path, "HAM10000_metadata.csv")
distribution_csv = "D:/Ascl_Mimic_Data/7_classes_on_3_Nodes.csv"
output_base = os.path.join(base_path, "Hospitals")

# Load data
metadata_df = pd.read_csv(metadata_path)
distribution_df = pd.read_csv(distribution_csv)

# Diagnosis label map
class_label_map = {
    "C_1": "bkl",    # Melanocytic nevi
    "C_2": "nv",   # Melanoma
    "C_3": "df",   # corrected Benign keratosis-like lesions
    "C_4": "mel",   # corrected Basal cell carcinoma
    "C_5": "vasc", # corrected Actinic keratoses
    "C_6": "bcc",  # Vascular lesions
    "C_7": "akiec"     # Dermatofibroma
}

# Ensure hospital directories exist
for node in distribution_df['Node'].unique():
    os.makedirs(os.path.join(output_base, node), exist_ok=True)

# Group image filenames by class label
class_to_images = defaultdict(list)
for _, row in metadata_df.iterrows():
    img_name = f"{row['image_id']}.jpg"
    class_label = row['dx']
    class_to_images[class_label].append(img_name)

# Sort each class list for consistency
for cls in class_to_images:
    class_to_images[cls].sort()

# Track how many images are already assigned per class
used_images = defaultdict(int)

# Distribute according to CSV
for _, row in distribution_df.iterrows():
    class_str = row['Class']         # E.g., C_1
    node = row['Node']               # Hosp1, Hosp2, etc.
    count = row['Data_Points']       # How many images to assign
    class_label = class_label_map[class_str]

    available_imgs = class_to_images[class_label]
    start = used_images[class_label]
    selected_imgs = available_imgs[start:start + count]

    for img in selected_imgs:
        # Look for image in part_1 or part_2
        src = os.path.join(image_folder_1, img)
        if not os.path.exists(src):
            src = os.path.join(image_folder_2, img)
        if not os.path.exists(src):
            print(f"❌ Image not found: {img}")
            continue

        dst_name = f"{class_str}_{img}"  # Prefix to keep unique
        dst = os.path.join(output_base, node, dst_name)
        shutil.copy2(src, dst)

    used_images[class_label] += count

print("✅ HAM10000 images distributed to Hosp1, Hosp2, and Hosp3.")
