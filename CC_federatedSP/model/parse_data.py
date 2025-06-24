import pandas as pd

def load_sipakmed_data(data_dir, metadata_csv='metadata.csv'):
    """Load data using a metadata CSV file."""
    # Read metadata
    df = pd.read_csv(os.path.join(data_dir, metadata_csv))
    
    # Define mapping from cellular types to class labels
    category_map = {
        'Normal': 0,
        'Abnormal': 1,
        'Benign': 2
    }
    
    # Alternative: Map cellular types directly
    Koilocytotic = 825
    Dyskeratotic = 813
    Metaplastic = 793
    Parabasal = 789
    Superficial = 831
    cellular_type_mapping = {
        'Koilocytotic': 1,    # Abnormal
        'Dyskeratotic': 1,    # Abnormal
        'Metaplastic': 2,     # Benign
        'Parabasal': 0,       # Normal
        'Superficial': 0,     # Normal
    }
    
    file_paths = []
    labels = []
    
    for _, row in df.iterrows():
        img_path = os.path.join(data_dir, row['image_filename'])
        if os.path.exists(img_path):
            # Option 1: Use existing category if available
            if 'category' in row:
                label = category_map[row['category']]
            # Option 2: Determine from cellular type
            else:
                label = cellular_type_mapping.get(row['cellular_type'], -1)
            
            if label != -1:
                file_paths.append(img_path)
                labels.append(label)
    
    return file_paths, labels