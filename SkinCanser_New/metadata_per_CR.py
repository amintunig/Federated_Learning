import pandas as pd
import os

# Load the big metadata
metadata = pd.read_csv("D:/Ascl_Mimic_Data/NEW_EXP/Ex1_metadata.csv")

# Function to split for a CR folder
def split_metadata(metadata, cr_folder, output_csv):
    # get filenames actually present in the folder
    image_files = set(os.listdir(cr_folder))
    
    # filter metadata to only those
    filtered = metadata[metadata['filepath'].apply(lambda x: os.path.basename(x) in image_files)]
    
    # fix filepath to be just the filename
    filtered['filepath'] = filtered['filepath'].apply(os.path.basename)
    
    # save
    filtered.to_csv(output_csv, index=False)
    print(f"Metadata saved to {output_csv} with {len(filtered)} entries.")

# call for each CR
split_metadata(metadata, 
               r"D:/Ascl_Mimic_Data/NEW_EXP/Ex1/CR_1", 
               r"D:/Ascl_Mimic_Data/NEW_EXP/Ex1_metadata_CR1.csv")

split_metadata(metadata, 
               r"D:/Ascl_Mimic_Data/NEW_EXP/Ex1/CR_2", 
               r"D:/Ascl_Mimic_Data/NEW_EXP/Ex1_metadata_CR2.csv")

split_metadata(metadata, 
               r"D:/Ascl_Mimic_Data/NEW_EXP/Ex1/CR_3", 
               r"D:/Ascl_Mimic_Data/NEW_EXP/Ex1_metadata_CR3.csv")
