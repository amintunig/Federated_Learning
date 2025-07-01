import os
import pandas as pd

# Paths
base_path = "D:/Ascl_Mimic_Data/SkinCancerMNIST"
metadata_path = os.path.join(base_path, "HAM10000_metadata.csv")
ex1_path = os.path.join(base_path, "Hospitals", "Ex1")
output_metadata_path = os.path.join(base_path, "Ex1_metadata.csv")  # Output CSV path

# Load metadata CSV
metadata_df = pd.read_csv(metadata_path)

# Convert image_id column to a set for fast lookup
metadata_image_ids = set(metadata_df["image_id"])

records = []

# Iterate over each hospital folder (CR_1, CR_2, CR_3, etc.) inside Ex1
for hospital_folder in os.listdir(ex1_path):
    hospital_folder_path = os.path.join(ex1_path, hospital_folder)
    if not os.path.isdir(hospital_folder_path):
        continue  # Skip if not a directory

    # Iterate over all files in this hospital folder
    for fname in os.listdir(hospital_folder_path):
        if not fname.lower().endswith(".jpg"):
            continue

        # Extract image_id from filename, e.g. "C_1_ISIC_0024306.jpg" -> "ISIC_0024306"
        parts = fname.split("_")
        image_id = None
        for idx, part in enumerate(parts):
            if part.startswith("ISIC"):
                image_id = "_".join(parts[idx:]).replace(".jpg", "")
                break

        if image_id is None:
            print(f"❌ Cannot extract image_id from filename: {fname}")
            continue

        # Check if image_id exists in metadata
        if image_id not in metadata_image_ids:
            print(f"❌ Image_id {image_id} from filename {fname} not found in metadata.")
            continue

        # Get metadata row
        match = metadata_df[metadata_df["image_id"] == image_id].iloc[0]

        # Append record with metadata and filepath info
        records.append({
            "image_id": match["image_id"],
            "lesion_id": match["lesion_id"],
            "dx": match["dx"],
            "dx_type": match.get("dx_type", ""),
            "age": match.get("age", ""),
            "sex": match.get("sex", ""),
            "localization": match.get("localization", ""),
            "filepath": os.path.join("Hospitals", "Ex1", hospital_folder, fname),
            "hospital": hospital_folder
        })

# Create DataFrame from records
df = pd.DataFrame(records)

# Save to CSV
df.to_csv(output_metadata_path, index=False)
print(f"✅ Metadata saved to {output_metadata_path} with {len(df)} entries.")
