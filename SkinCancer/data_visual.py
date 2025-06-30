import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class FederatedDataPartitioner:
    """
    A comprehensive class for analyzing federated learning data partition scenarios
    for the SkinCancerMNIST dataset.
    """
    
    def __init__(self):
        # Original dataset statistics
        self.total_samples = 10016
        self.class_distribution = {
            'c1': {'count': 1099, 'percentage': 10.97},
            'c2': {'count': 6705, 'percentage': 66.95},
            'c3': {'count': 115, 'percentage': 1.15},
            'c4': {'count': 1113, 'percentage': 11.11},
            'c5': {'count': 142, 'percentage': 1.42},
            'c6': {'count': 514, 'percentage': 5.13},
            'c7': {'count': 327, 'percentage': 3.27}
        }
        
        # Define partition scenarios
        self.scenarios = self._define_scenarios()
        
    def _define_scenarios(self) -> Dict:
        """Define the four data partition scenarios."""
        return {
            'balanced_balanced': {
                'name': 'Statistically and Class Balanced',
                'description': 'Equal data distribution with balanced class representation',
                'node1': {'total': 5005, 'percentage': 50},
                'node2': {'total': 5011, 'percentage': 50},
                'class_dist': {
                    'node1': {'c1': 549, 'c2': 3352, 'c3': 57, 'c4': 556, 'c5': 71, 'c6': 257, 'c7': 163},
                    'node2': {'c1': 550, 'c2': 3353, 'c3': 58, 'c4': 557, 'c5': 71, 'c6': 257, 'c7': 164}
                },
                'characteristics': [
                    'Equal data distribution between nodes',
                    'Proportional class representation',
                    'Most stable federated learning scenario',
                    'Minimal statistical heterogeneity'
                ]
            },
            'balanced_unbalanced': {
                'name': 'Statistically Balanced and Class Unbalanced',
                'description': 'Equal data amounts but skewed class distributions',
                'node1': {'total': 4003, 'percentage': 40},
                'node2': {'total': 6013, 'percentage': 60},
                'class_dist': {
                    'node1': {'c1': 439, 'c2': 2682, 'c3': 46, 'c4': 445, 'c5': 56, 'c6': 205, 'c7': 130},
                    'node2': {'c1': 660, 'c2': 4023, 'c3': 69, 'c4': 668, 'c5': 86, 'c6': 309, 'c7': 197}
                },
                'characteristics': [
                    'Unequal statistical distribution',
                    'Balanced class representation per node',
                    'Moderate heterogeneity',
                    'Challenges in model convergence'
                ]
            },
            'unbalanced_balanced': {
                'name': 'Statistically Unbalanced and Class Balanced',
                'description': 'Unequal data amounts but proportional class distributions',
                'node1': {'total': 3001, 'percentage': 30},
                'node2': {'total': 7015, 'percentage': 70},
                'class_dist': {
                    'node1': {'c1': 329, 'c2': 2011, 'c3': 34, 'c4': 333, 'c5': 42, 'c6': 154, 'c7': 98},
                    'node2': {'c1': 770, 'c2': 4694, 'c3': 81, 'c4': 780, 'c5': 100, 'c6': 360, 'c7': 229}
                },
                'characteristics': [
                    'Unequal data distribution',
                    'Proportional class distributions',
                    'Node capacity heterogeneity',
                    'Potential training imbalance'
                ]
            },
            'unbalanced_unbalanced': {
                'name': 'Statistically Unbalanced and Class Unbalanced',
                'description': 'Both unequal data amounts and skewed class distributions',
                'node1': {'total': 2000, 'percentage': 20},
                'node2': {'total': 8016, 'percentage': 80},
                'class_dist': {
                    'node1': {'c1': 219, 'c2': 1341, 'c3': 23, 'c4': 222, 'c5': 28, 'c6': 102, 'c7': 65},
                    'node2': {'c1': 880, 'c2': 5364, 'c3': 92, 'c4': 891, 'c5': 114, 'c6': 412, 'c7': 262}
                },
                'characteristics': [
                    'Maximum heterogeneity',
                    'Non-IID data distribution',
                    'Significant convergence challenges',
                    'Requires advanced FL algorithms'
                ]
            }
        }
    
    def calculate_heterogeneity_metrics(self, scenario_key: str) -> Dict:
        """Calculate heterogeneity metrics for a given scenario."""
        scenario = self.scenarios[scenario_key]
        
        # Statistical heterogeneity (difference in data amounts)
        node1_total = scenario['node1']['total']
        node2_total = scenario['node2']['total']
        stat_heterogeneity = abs(node1_total - node2_total) / max(node1_total, node2_total)
        
        # Class heterogeneity (KL divergence between class distributions)
        node1_dist = np.array([scenario['class_dist']['node1'][f'c{i}'] for i in range(1, 8)])
        node2_dist = np.array([scenario['class_dist']['node2'][f'c{i}'] for i in range(1, 8)])
        
        # Normalize to probabilities
        node1_prob = node1_dist / node1_dist.sum()
        node2_prob = node2_dist / node2_dist.sum()
        
        # Calculate KL divergence
        kl_div = np.sum(node1_prob * np.log(node1_prob / (node2_prob + 1e-10)))
        
        return {
            'statistical_heterogeneity': stat_heterogeneity,
            'class_heterogeneity_kl': kl_div,
            'node1_samples': node1_total,
            'node2_samples': node2_total,
            'ratio': node1_total / node2_total
        }
    
    def create_comparison_dataframe(self) -> pd.DataFrame:
        """Create a comprehensive comparison dataframe."""
        data = []
        
        for scenario_key, scenario in self.scenarios.items():
            metrics = self.calculate_heterogeneity_metrics(scenario_key)
            
            data.append({
                'Scenario': scenario['name'],
                'Node1_Samples': metrics['node1_samples'],
                'Node2_Samples': metrics['node2_samples'],
                'Sample_Ratio': metrics['ratio'],
                'Statistical_Heterogeneity': metrics['statistical_heterogeneity'],
                'Class_Heterogeneity_KL': metrics['class_heterogeneity_kl'],
                'Description': scenario['description']
            })
        
        return pd.DataFrame(data)
    
    def plot_scenario_overview(self, figsize=(20, 15)):
        """Create a comprehensive visualization of all scenarios."""
        fig = plt.figure(figsize=figsize)
        
        # Create subplots
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        scenarios_list = list(self.scenarios.items())
        colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6']
        
        for i, (scenario_key, scenario) in enumerate(scenarios_list):
            # Data distribution plot
            ax1 = fig.add_subplot(gs[0, i])
            nodes = ['Node 1', 'Node 2']
            samples = [scenario['node1']['total'], scenario['node2']['total']]
            bars = ax1.bar(nodes, samples, color=[colors[i], colors[i]], alpha=0.7)
            ax1.set_title(f'{scenario["name"]}\nData Distribution', fontsize=10, fontweight='bold')
            ax1.set_ylabel('Number of Samples')
            
            # Add value labels on bars
            for bar, sample in zip(bars, samples):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
                        f'{sample:,}', ha='center', va='bottom', fontsize=9)
            
            # Class distribution plot
            ax2 = fig.add_subplot(gs[1, i])
            classes = [f'C{j}' for j in range(1, 8)]
            node1_classes = [scenario['class_dist']['node1'][f'c{j}'] for j in range(1, 8)]
            node2_classes = [scenario['class_dist']['node2'][f'c{j}'] for j in range(1, 8)]
            
            x = np.arange(len(classes))
            width = 0.35
            
            ax2.bar(x - width/2, node1_classes, width, label='Node 1', alpha=0.7, color=colors[i])
            ax2.bar(x + width/2, node2_classes, width, label='Node 2', alpha=0.7, color='gray')
            
            ax2.set_title('Class Distribution', fontsize=10, fontweight='bold')
            ax2.set_xlabel('Classes')
            ax2.set_ylabel('Number of Samples')
            ax2.set_xticks(x)
            ax2.set_xticklabels(classes)
            ax2.legend(fontsize=8)
            ax2.tick_params(axis='x', labelsize=8)
            ax2.tick_params(axis='y', labelsize=8)
        
        # Heterogeneity comparison
        ax3 = fig.add_subplot(gs[2, :2])
        df = self.create_comparison_dataframe()
        
        x = np.arange(len(df))
        width = 0.35
        
        ax3.bar(x - width/2, df['Statistical_Heterogeneity'], width, 
                label='Statistical Heterogeneity', alpha=0.7, color='#3498db')
        ax3.bar(x + width/2, df['Class_Heterogeneity_KL'], width,
                label='Class Heterogeneity (KL)', alpha=0.7, color='#e74c3c')
        
        ax3.set_title('Heterogeneity Metrics Comparison', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Scenarios')
        ax3.set_ylabel('Heterogeneity Score')
        ax3.set_xticks(x)
        ax3.set_xticklabels([s.replace(' ', '\n') for s in df['Scenario']], fontsize=9)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Sample ratio comparison
        ax4 = fig.add_subplot(gs[2, 2:])
        bars = ax4.bar(range(len(df)), df['Sample_Ratio'], color=colors, alpha=0.7)
        ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Perfect Balance')
        ax4.set_title('Node Sample Ratio (Node1/Node2)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Scenarios')
        ax4.set_ylabel('Ratio')
        ax4.set_xticks(range(len(df)))
        ax4.set_xticklabels([s.replace(' ', '\n') for s in df['Scenario']], fontsize=9)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, ratio in zip(bars, df['Sample_Ratio']):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{ratio:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Federated Learning Data Partition Scenarios - SkinCancerMNIST Dataset', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.show()
    
    def plot_detailed_class_analysis(self, figsize=(16, 12)):
        """Create detailed class distribution analysis."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        scenarios_list = list(self.scenarios.items())
        colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6']
        
        for i, (scenario_key, scenario) in enumerate(scenarios_list):
            ax = axes[i]
            
            # Prepare data for stacked bar chart
            classes = [f'C{j}' for j in range(1, 8)]
            node1_data = [scenario['class_dist']['node1'][f'c{j}'] for j in range(1, 8)]
            node2_data = [scenario['class_dist']['node2'][f'c{j}'] for j in range(1, 8)]
            
            # Create stacked bar chart
            width = 0.6
            x = np.arange(len(classes))
            
            p1 = ax.bar(x, node1_data, width, label='Node 1', color=colors[i], alpha=0.8)
            p2 = ax.bar(x, node2_data, width, bottom=node1_data, label='Node 2', 
                       color='lightgray', alpha=0.8)
            
            ax.set_title(scenario['name'], fontsize=12, fontweight='bold')
            ax.set_xlabel('Classes')
            ax.set_ylabel('Number of Samples')
            ax.set_xticks(x)
            ax.set_xticklabels(classes)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add percentage labels
            total_samples = [n1 + n2 for n1, n2 in zip(node1_data, node2_data)]
            for j, (n1, n2, total) in enumerate(zip(node1_data, node2_data, total_samples)):
                if total > 0:
                    ax.text(j, n1/2, f'{n1/total*100:.1f}%', 
                           ha='center', va='center', fontsize=8, fontweight='bold')
                    ax.text(j, n1 + n2/2, f'{n2/total*100:.1f}%', 
                           ha='center', va='center', fontsize=8, fontweight='bold')
        
        plt.suptitle('Detailed Class Distribution Analysis Across Scenarios', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def generate_statistics_table(self) -> pd.DataFrame:
        """Generate comprehensive statistics table."""
        data = []
        
        for scenario_key, scenario in self.scenarios.items():
            metrics = self.calculate_heterogeneity_metrics(scenario_key)
            
            # Calculate class-wise statistics
            for class_key in ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']:
                node1_count = scenario['class_dist']['node1'][class_key]
                node2_count = scenario['class_dist']['node2'][class_key]
                original_count = self.class_distribution[class_key]['count']
                
                data.append({
                    'Scenario': scenario['name'],
                    'Class': class_key.upper(),
                    'Original_Count': original_count,
                    'Node1_Count': node1_count,
                    'Node2_Count': node2_count,
                    'Node1_Percentage': (node1_count / (node1_count + node2_count)) * 100,
                    'Node2_Percentage': (node2_count / (node1_count + node2_count)) * 100,
                    'Distribution_Ratio': node1_count / node2_count if node2_count > 0 else float('inf')
                })
        
        return pd.DataFrame(data)
    
    def analyze_fl_impact(self):
        """Analyze the impact of each scenario on federated learning."""
        print("=" * 80)
        print("FEDERATED LEARNING IMPACT ANALYSIS")
        print("=" * 80)
        
        impact_analysis = {
            'balanced_balanced': {
                'convergence_speed': 'Fast',
                'communication_overhead': 'Low',
                'model_accuracy': 'High',
                'recommended_algorithm': 'FedAvg',
                'challenges': 'Minimal'
            },
            'balanced_unbalanced': {
                'convergence_speed': 'Moderate',
                'communication_overhead': 'Medium',
                'model_accuracy': 'Medium-High',
                'recommended_algorithm': 'FedAvg with weighted aggregation',
                'challenges': 'Statistical heterogeneity'
            },
            'unbalanced_balanced': {
                'convergence_speed': 'Moderate',
                'communication_overhead': 'Medium',
                'model_accuracy': 'Medium',
                'recommended_algorithm': 'FedAvg with sample weighting',
                'challenges': 'Unequal participation'
            },
            'unbalanced_unbalanced': {
                'convergence_speed': 'Slow',
                'communication_overhead': 'High',
                'model_accuracy': 'Low-Medium',
                'recommended_algorithm': 'FedProx, SCAFFOLD, or FedNova',
                'challenges': 'Maximum heterogeneity, client drift'
            }
        }
        
        for scenario_key, scenario in self.scenarios.items():
            impact = impact_analysis[scenario_key]
            metrics = self.calculate_heterogeneity_metrics(scenario_key)
            
            print(f"\n{scenario['name'].upper()}")
            print("-" * len(scenario['name']))
            print(f"Description: {scenario['description']}")
            print(f"Statistical Heterogeneity: {metrics['statistical_heterogeneity']:.3f}")
            print(f"Class Heterogeneity (KL): {metrics['class_heterogeneity_kl']:.3f}")
            print(f"Sample Ratio (N1/N2): {metrics['ratio']:.3f}")
            print(f"Convergence Speed: {impact['convergence_speed']}")
            print(f"Communication Overhead: {impact['communication_overhead']}")
            print(f"Expected Model Accuracy: {impact['model_accuracy']}")
            print(f"Recommended Algorithm: {impact['recommended_algorithm']}")
            print(f"Main Challenges: {impact['challenges']}")
            
            print("\nCharacteristics:")
            for char in scenario['characteristics']:
                print(f"  • {char}")
    
    def save_results(self, filename_prefix='federated_analysis'):
        """Save analysis results to files."""
        # Save statistics table
        stats_df = self.generate_statistics_table()
        stats_df.to_csv(f'{filename_prefix}_statistics.csv', index=False)
        
        # Save comparison table
        comparison_df = self.create_comparison_dataframe()
        comparison_df.to_csv(f'{filename_prefix}_comparison.csv', index=False)
        
        print(f"Results saved to {filename_prefix}_statistics.csv and {filename_prefix}_comparison.csv")

# Example usage and demonstration
def main():
    """Main function to demonstrate the federated data partitioner."""
    
    # Initialize the partitioner
    partitioner = FederatedDataPartitioner()
    
    # Display analysis
    partitioner.analyze_fl_impact()
    
    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS...")
    print("="*80)
    
    # Plot comprehensive scenario overview
    partitioner.plot_scenario_overview()
    
    # Plot detailed class analysis
    partitioner.plot_detailed_class_analysis()
    
    # Display statistics table
    print("\n" + "="*80)
    print("DETAILED STATISTICS TABLE")
    print("="*80)
    
    stats_df = partitioner.generate_statistics_table()
    print(stats_df.to_string(index=False))
    
    # Display comparison table
    print("\n" + "="*80)
    print("SCENARIO COMPARISON TABLE")
    print("="*80)
    
    comparison_df = partitioner.create_comparison_dataframe()
    print(comparison_df.to_string(index=False))
    
    # Save results
    partitioner.save_results()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()