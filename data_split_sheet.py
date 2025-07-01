import os
import shutil
import pandas as pd
from collections import defaultdict

# --- Paths ---
base_path = "D:/Ascl_Mimic_Data/SkinCancerMNIST"
image_folder_1 = os.path.join(base_path, "HAM10000_images_part_1")
image_folder_2 = os.path.join(base_path, "HAM10000_images_part_2")
metadata_path = os.path.join(base_path, "HAM10000_metadata.csv")
distribution_excel = os.path.join("D:/Ascl_Mimic_Data/data_per_sheets.xlsx")
output_base = os.path.join(base_path, "Hospitals")

# --- Load metadata ---
metadata_df = pd.read_csv(metadata_path)

# --- Map class codes to actual dx labels ---
# class_label_map = {
#     "C_1": "nv", "C_2": "mel", "C_3": "bkl", "C_4": "bcc",
#     "C_5": "akiec", "C_6": "vasc", "C_7": "df"
# }
class_label_map = {"C_1": "bkl", "C_2": "nv", "C_3": "df", "C_4": "mel",   
    "C_5": "vasc", "C_6": "bcc",  "C_7": "akiec"     
}

# --- Collect images by class ---
class_to_images = defaultdict(list)
for _, row in metadata_df.iterrows():
    img_name = f"{row['image_id']}.jpg"
    class_label = row['dx']
    class_to_images[class_label].append(img_name)

# Sort for consistency
for cls in class_to_images:
    class_to_images[cls].sort()

# --- Load Excel sheets ---
excel_data = pd.read_excel(distribution_excel, sheet_name=None, engine='openpyxl')

# --- Process each sheet ---
for sheet_name, distribution_df in excel_data.items():
    print(f"\n📄 Processing sheet: {sheet_name}")
    sheet_folder = os.path.join(output_base, sheet_name)
    os.makedirs(sheet_folder, exist_ok=True)

    used_images = defaultdict(set)
    total_requested = 0
    total_copied = 0
    total_missing = 0

    for _, row in distribution_df.iterrows():
        class_code = row["Class"]
        node = row["Node"]
        requested = int(row["Data_Points"])
        total_requested += requested

        class_label = class_label_map[class_code]

        # Filter unused images
        available_imgs = [img for img in class_to_images[class_label] if img not in used_images[class_label]]

        if len(available_imgs) < requested:
            print(f"⚠️ Not enough images for class {class_label} in {sheet_name}/{node}: Requested {requested}, available {len(available_imgs)}")
            requested = len(available_imgs)

        selected_imgs = available_imgs[:requested]
        used_images[class_label].update(selected_imgs)

        # Create node folder
        node_folder = os.path.join(sheet_folder, node)
        os.makedirs(node_folder, exist_ok=True)

        copied_now = 0

        for img in selected_imgs:
            src = os.path.join(image_folder_1, img)
            if not os.path.exists(src):
                src = os.path.join(image_folder_2, img)

            if not os.path.exists(src):
                print(f"❌ Missing image: {img}")
                total_missing += 1
                continue

            dst_name = f"{class_code}_{img}"
            dst = os.path.join(node_folder, dst_name)
            shutil.copy2(src, dst)
            copied_now += 1

        total_copied += copied_now

    # ✅ Summary for this sheet
    print(f"\n📊 Summary for {sheet_name}:")
    print(f"    📦 Total requested images: {total_requested}")
    print(f"    ✅ Total copied images:    {total_copied}")
    print(f"    ❌ Missing images:         {total_missing}")
    print(f"    💾 Stored in folder:       {sheet_folder}")

print("\n✅ Distribution complete for all sheets.")
