import os
import random
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

def load_sipakmed_data(data_dir):
    classes = ["Normal", "Abnormal", "Benign"]
    file_paths, labels = [], []
    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for root, _, files in os.walk(class_dir):
            for fname in files:
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
                    file_paths.append(os.path.join(root, fname))
                    labels.append(label_idx)
    return file_paths, labels

def partition_data(file_paths, labels, num_clients=3, scenario=1, seed=42):
    random.seed(seed)
    indices_by_class = {}
    for idx, label in enumerate(labels):
        indices_by_class.setdefault(label, []).append(idx)

    for idxs in indices_by_class.values():
        random.shuffle(idxs)
        
    client_data = {i: {'file_paths': [], 'labels': []} for i in range(num_clients)}
    total = len(labels)
    if scenario == 1:
        # Stats & class balanced: split each class evenly
        for cls, idxs in indices_by_class.items():
            n = len(idxs)
            base = n // num_clients
            rem = n - base * (num_clients - 1)
            splits = [base] * (num_clients - 1) + [rem]
            start = 0
            for i, count in enumerate(splits):
                end = start + count
                for idx in idxs[start:end]:
                    client_data[i]['file_paths'].append(file_paths[idx])
                    client_data[i]['labels'].append(labels[idx])
                start = end
    elif scenario == 2:
        # Stats balanced, class unbalanced: custom fractions per class
        fracs = {
            0: [0.494, 0.124, 0.382],  # Normal
            1: [0.244, 0.611, 0.145],  # Abnormal
            2: [0.188, 0.189, 0.623]   # Benign
        }
        for cls, idxs in indices_by_class.items():
            n = len(idxs)
            frac = fracs.get(cls, [1/num_clients]*num_clients)
            counts = [int(round(n * f)) for f in frac]
            counts[-1] = n - sum(counts[:-1])  # adjust last to match total
            start = 0
            for i, count in enumerate(counts):
                end = start + count
                for idx in idxs[start:end]:
                    client_data[i]['file_paths'].append(file_paths[idx])
                    client_data[i]['labels'].append(labels[idx])
                start = end
    elif scenario == 3:
        # Stats unbalanced, class balanced: allocate by global class proportions
        client_sizes = [int(total * 0.445), int(total * 0.372)]
        client_sizes.append(total - sum(client_sizes))
        frac_class = {cls: len(idxs)/total for cls, idxs in indices_by_class.items()}
        allocated = {cls: 0 for cls in indices_by_class}
        counts_matrix = []
        for size in client_sizes[:-1]:
            row = {}
            for cls, idxs in indices_by_class.items():
                cnt = int(round(size * frac_class[cls]))
                row[cls] = cnt
                allocated[cls] += cnt
            counts_matrix.append(row)
        row_last = {cls: len(indices_by_class[cls]) - allocated[cls] for cls in indices_by_class}
        counts_matrix.append(row_last)
        for cls, idxs in indices_by_class.items():
            start = 0
            for i, row in enumerate(counts_matrix):
                count = row[cls]
                end = start + count
                for idx in idxs[start:end]:
                    client_data[i]['file_paths'].append(file_paths[idx])
                    client_data[i]['labels'].append(labels[idx])
                start = end
    elif scenario == 4:
        # Stats & class unbalanced: extreme split, custom fractions
        fracs = {
            0: [0.618, 0.247, 0.135],
            1: [0.305, 0.611, 0.084],
            2: [0.630, 0.252, 0.117]
        }
        for cls, idxs in indices_by_class.items():
            n = len(idxs)
            frac = fracs.get(cls, [1/num_clients]*num_clients)
            counts = [int(round(n * f)) for f in frac]
            counts[-1] = n - sum(counts[:-1])
            start = 0
            for i, count in enumerate(counts):
                end = start + count
                for idx in idxs[start:end]:
                    client_data[i]['file_paths'].append(file_paths[idx])
                    client_data[i]['labels'].append(labels[idx])
                start = end
    # Shuffle each client's data for randomness
    for i in range(num_clients):
        combined = list(zip(client_data[i]['file_paths'], client_data[i]['labels']))
        random.shuffle(combined)
        if combined:
            paths, labs = zip(*combined)
            client_data[i]['file_paths'], client_data[i]['labels'] = list(paths), list(labs)
        else:
            client_data[i]['file_paths'], client_data[i]['labels'] = [], []
    return {i: (client_data[i]['file_paths'], client_data[i]['labels']) for i in range(num_clients)}

class SipaKMedDataset(Dataset):
    """
    PyTorch Dataset for SIPaKMeD images.
    """
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label
