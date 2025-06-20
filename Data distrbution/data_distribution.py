import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from typing import Dict, List, Tuple

class SipaKMedFLDistributor:
    def __init__(self, dataset_path: str, num_clients: int = 4):
        self.dataset_path = dataset_path
        self.num_clients = num_clients
        self.cell_classes = ['Koilocytotic', 'Metaplastic', 'Dyskeratotic', 'Parabasal', 'Superficial-Intermediate']
        self.disease_classes = ['normal', 'abnormal', 'benign']
        self.data_distribution = {}
        
    def load_dataset_info(self):
        """Load and organize dataset information"""
        # Simulated dataset structure - replace with your actual loading logic
        dataset_info = []
        class_counts = {
            'Koilocytotic': 825,
            'Metaplastic': 793,
            'Dyskeratotic': 813,
            'Parabasal': 787,
            'Superficial-Intermediate': 831
        }
        
        sample_id = 0
        for cell_class, count in class_counts.items():
            # Distribute disease classes within each cell class
            disease_per_cell = count // 3
            remainder = count % 3
            
            for i, disease in enumerate(self.disease_classes):
                disease_count = disease_per_cell + (1 if i < remainder else 0)
                for j in range(disease_count):
                    dataset_info.append({
                        'sample_id': sample_id,
                        'cell_class': cell_class,
                        'disease_class': disease,
                        'combined_class': f"{cell_class}_{disease}",
                        'file_path': f"{cell_class}/{disease}/sample_{sample_id}.jpg"
                    })
                    sample_id += 1
        
        self.dataset_df = pd.DataFrame(dataset_info)
        return self.dataset_df
    
    def scenario_1_stat_balanced_class_balanced(self) -> Dict[int, List]:
        """Statistically balanced and class balanced distribution"""
        print("Scenario 1: Statistically Balanced + Class Balanced")
        
        client_data = {i: [] for i in range(self.num_clients)}
        
        # Group by combined classes
        combined_classes = self.dataset_df['combined_class'].unique()
        samples_per_client_per_class = len(self.dataset_df) // (self.num_clients * len(combined_classes))
        
        for combined_class in combined_classes:
            class_samples = self.dataset_df[self.dataset_df['combined_class'] == combined_class].sample(frac=1).reset_index(drop=True)
            
            # Distribute equally across clients
            for client_id in range(self.num_clients):
                start_idx = client_id * samples_per_client_per_class
                end_idx = min((client_id + 1) * samples_per_client_per_class, len(class_samples))
                client_data[client_id].extend(class_samples.iloc[start_idx:end_idx]['sample_id'].tolist())
        
        return client_data
    
    def scenario_2_stat_balanced_class_unbalanced(self) -> Dict[int, List]:
        """Statistically balanced and class unbalanced distribution"""
        print("Scenario 2: Statistically Balanced + Class Unbalanced")
        
        client_data = {i: [] for i in range(self.num_clients)}
        total_samples = len(self.dataset_df)
        samples_per_client = total_samples // self.num_clients
        
        # Create imbalanced class distribution but balanced client sizes
        combined_classes = self.dataset_df['combined_class'].unique()
        
        # Define imbalanced ratios for each client
        class_ratios = [
            [0.4, 0.3, 0.15, 0.1, 0.05],  # Client 0: heavily skewed
            [0.1, 0.4, 0.3, 0.15, 0.05],  # Client 1: different skew
            [0.05, 0.1, 0.4, 0.3, 0.15],  # Client 2: different skew
            [0.15, 0.05, 0.1, 0.3, 0.4]   # Client 3: opposite skew
        ]
        
        for client_id in range(self.num_clients):
            client_samples = []
            ratios = class_ratios[client_id % len(class_ratios)]
            
            for i, combined_class in enumerate(combined_classes[:5]):  # Use first 5 classes
                class_samples = self.dataset_df[self.dataset_df['combined_class'] == combined_class]
                num_samples = int(samples_per_client * ratios[i])
                selected_samples = class_samples.sample(n=min(num_samples, len(class_samples)))
                client_samples.extend(selected_samples['sample_id'].tolist())
            
            # Fill remaining slots randomly if needed
            remaining_slots = samples_per_client - len(client_samples)
            if remaining_slots > 0:
                remaining_samples = self.dataset_df[~self.dataset_df['sample_id'].isin(client_samples)]
                additional_samples = remaining_samples.sample(n=min(remaining_slots, len(remaining_samples)))
                client_samples.extend(additional_samples['sample_id'].tolist())
            
            client_data[client_id] = client_samples[:samples_per_client]
        
        return client_data
    
    def scenario_3_stat_unbalanced_class_balanced(self) -> Dict[int, List]:
        """Statistically unbalanced and class balanced distribution"""
        print("Scenario 3: Statistically Unbalanced + Class Balanced")
        
        client_data = {i: [] for i in range(self.num_clients)}
        
        # Define unbalanced client sizes
        total_samples = len(self.dataset_df)
        client_ratios = [0.4, 0.3, 0.2, 0.1]  # Unbalanced client sizes
        
        combined_classes = self.dataset_df['combined_class'].unique()
        
        for client_id in range(self.num_clients):
            client_size = int(total_samples * client_ratios[client_id])
            samples_per_class = client_size // len(combined_classes)
            
            client_samples = []
            for combined_class in combined_classes:
                class_samples = self.dataset_df[self.dataset_df['combined_class'] == combined_class]
                selected_samples = class_samples.sample(n=min(samples_per_class, len(class_samples)))
                client_samples.extend(selected_samples['sample_id'].tolist())
            
            client_data[client_id] = client_samples[:client_size]
        
        return client_data
    
    def scenario_4_stat_unbalanced_class_unbalanced(self) -> Dict[int, List]:
        """Statistically unbalanced and class unbalanced distribution"""
        print("Scenario 4: Statistically Unbalanced + Class Unbalanced")
        
        client_data = {i: [] for i in range(self.num_clients)}
        
        # Define unbalanced client sizes
        total_samples = len(self.dataset_df)
        client_size_ratios = [0.4, 0.3, 0.2, 0.1]
        
        # Define different class preferences for each client
        class_preferences = [
            {'Koilocytotic': 0.5, 'Metaplastic': 0.3, 'Dyskeratotic': 0.2},
            {'Metaplastic': 0.4, 'Parabasal': 0.4, 'Superficial-Intermediate': 0.2},
            {'Dyskeratotic': 0.6, 'Koilocytotic': 0.3, 'Parabasal': 0.1},
            {'Superficial-Intermediate': 0.7, 'Metaplastic': 0.3}
        ]
        
        for client_id in range(self.num_clients):
            client_size = int(total_samples * client_size_ratios[client_id])
            preferences = class_preferences[client_id]
            
            client_samples = []
            for cell_class, ratio in preferences.items():
                target_samples = int(client_size * ratio)
                class_data = self.dataset_df[self.dataset_df['cell_class'] == cell_class]
                selected_samples = class_data.sample(n=min(target_samples, len(class_data)))
                client_samples.extend(selected_samples['sample_id'].tolist())
            
            client_data[client_id] = client_samples[:client_size]
        
        return client_data
    
    def analyze_distribution(self, client_data: Dict[int, List], scenario_name: str):
        """Analyze and visualize the distribution"""
        print(f"\n=== {scenario_name} Analysis ===")
        
        # Statistical analysis
        client_sizes = [len(samples) for samples in client_data.values()]
        print(f"Client sizes: {client_sizes}")
        print(f"Size variance: {np.var(client_sizes):.2f}")
        
        # Class distribution analysis
        for client_id, sample_ids in client_data.items():
            client_samples = self.dataset_df[self.dataset_df['sample_id'].isin(sample_ids)]
            class_dist = client_samples['combined_class'].value_counts()
            print(f"Client {client_id}: {len(sample_ids)} samples")
            print(f"  Class distribution: {dict(class_dist.head())}")
        
        return client_data
    
    def visualize_distribution(self, client_data: Dict[int, List], scenario_name: str):
        """Create visualization of the distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"{scenario_name} - Distribution Analysis", fontsize=16)
        
        # Client size distribution
        client_sizes = [len(samples) for samples in client_data.values()]
        axes[0, 0].bar(range(self.num_clients), client_sizes)
        axes[0, 0].set_title("Client Size Distribution")
        axes[0, 0].set_xlabel("Client ID")
        axes[0, 0].set_ylabel("Number of Samples")
        
        # Class distribution per client
        combined_classes = self.dataset_df['combined_class'].unique()
        class_matrix = np.zeros((self.num_clients, len(combined_classes)))
        
        for client_id, sample_ids in client_data.items():
            client_samples = self.dataset_df[self.dataset_df['sample_id'].isin(sample_ids)]
            class_counts = client_samples['combined_class'].value_counts()
            for i, class_name in enumerate(combined_classes):
                class_matrix[client_id, i] = class_counts.get(class_name, 0)
        
        im = axes[0, 1].imshow(class_matrix, cmap='Blues', aspect='auto')
        axes[0, 1].set_title("Class Distribution Heatmap")
        axes[0, 1].set_xlabel("Combined Classes")
        axes[0, 1].set_ylabel("Client ID")
        plt.colorbar(im, ax=axes[0, 1])
        
        # Cell class distribution
        for client_id, sample_ids in client_data.items():
            client_samples = self.dataset_df[self.dataset_df['sample_id'].isin(sample_ids)]
            cell_dist = client_samples['cell_class'].value_counts()
            axes[1, client_id % 2].pie(cell_dist.values, labels=cell_dist.index, autopct='%1.1f%%')
            axes[1, client_id % 2].set_title(f"Client {client_id} Cell Class Distribution")
        
        plt.tight_layout()
        plt.savefig(f"{scenario_name}_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_all_scenarios(self):
        """Run all four distribution scenarios"""
        self.load_dataset_info()
        
        scenarios = [
            ("Scenario 1: Stat Balanced + Class Balanced", self.scenario_1_stat_balanced_class_balanced),
            ("Scenario 2: Stat Balanced + Class Unbalanced", self.scenario_2_stat_balanced_class_unbalanced),
            ("Scenario 3: Stat Unbalanced + Class Balanced", self.scenario_3_stat_unbalanced_class_balanced),
            ("Scenario 4: Stat Unbalanced + Class Unbalanced", self.scenario_4_stat_unbalanced_class_unbalanced)
        ]
        
        results = {}
        for scenario_name, scenario_func in scenarios:
            client_data = scenario_func()
            analyzed_data = self.analyze_distribution(client_data, scenario_name)
            self.visualize_distribution(client_data, scenario_name)
            results[scenario_name] = analyzed_data
        
        return results

# Usage example
def main():
    # Initialize the distributor
    distributor = SipaKMedFLDistributor(
        dataset_path="D:/Ascl_Mimic_Data/CC_Kaggle_Datasets",
        num_clients=4
    )
    
    # Run all scenarios
    all_distributions = distributor.run_all_scenarios()
    
    # Save distributions for federated learning
    for scenario_name, client_data in all_distributions.items():
        np.save(f"{scenario_name.replace(' ', '_').replace(':', '')}_distribution.npy", client_data)
        print(f"Saved {scenario_name} distribution")

if __name__ == "__main__":
    main()
