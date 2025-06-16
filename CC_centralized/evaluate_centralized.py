""" 
Evaluation script for Centralized Training 
Contains evaluation utilities and metrics calculation for centralized model 
""" 

import torch 
import torch.nn as nn 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import ( 
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score, roc_curve 
) 
from sklearn.preprocessing import label_binarize 
from typing import Dict, List, Tuple, Optional 
import logging 
import json 
import os 
from torch.utils.data import DataLoader 


class MetricsCalculator: 
    """Calculate and track various metrics for model evaluation""" 
    
    def __init__(self, num_classes: int = 5, class_names: Optional[List[str]] = None): 
        self.num_classes = num_classes 
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)] 
        self.reset() 
    
    def reset(self): 
        """Reset all metrics""" 
        self.all_predictions = [] 
        self.all_targets = [] 
        self.all_probabilities = [] 
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor,  
               probabilities: Optional[torch.Tensor] = None): 
        """Update metrics with new batch""" 
        # Convert to CPU and numpy 
        pred_np = predictions.cpu().numpy() 
        target_np = targets.cpu().numpy() 
        
        self.all_predictions.extend(pred_np) 
        self.all_targets.extend(target_np) 
        
        if probabilities is not None: 
            prob_np = probabilities.cpu().numpy() 
            self.all_probabilities.extend(prob_np) 
    
    def compute_metrics(self) -> Dict[str, float]: 
        """Compute all metrics""" 
        y_true = np.array(self.all_targets) 
        y_pred = np.array(self.all_predictions) 
        
        metrics = {} 
        
        # Basic metrics 
        metrics['accuracy'] = accuracy_score(y_true, y_pred) 
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0) 
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0) 
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0) 
        
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0) 
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0) 
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0) 
        
        # Per-class metrics 
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0) 
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0) 
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0) 
        
        for i, class_name in enumerate(self.class_names): 
            metrics[f'precision_{class_name}'] = precision_per_class[i] 
            metrics[f'recall_{class_name}'] = recall_per_class[i] 
            metrics[f'f1_{class_name}'] = f1_per_class[i] 
        
        # ROC AUC if probabilities are available 
        if self.all_probabilities: 
            try: 
                y_prob = np.array(self.all_probabilities) 
                # Binarize labels for multiclass ROC AUC 
                y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes))) 
                
                if self.num_classes == 2: 
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1]) 
                else: 
                    metrics['roc_auc_macro'] = roc_auc_score(y_true_bin, y_prob,  
                                                           average='macro', multi_class='ovr') 
                    metrics['roc_auc_micro'] = roc_auc_score(y_true_bin, y_prob,  
                                                           average='micro', multi_class='ovr') 
            except Exception as e: 
                logging.warning(f"Could not compute ROC AUC: {e}") 
        
        return metrics 
    
    def get_confusion_matrix(self) -> np.ndarray: 
        """Get confusion matrix""" 
        y_true = np.array(self.all_targets) 
        y_pred = np.array(self.all_predictions) 
        return confusion_matrix(y_true, y_pred) 
    
    def plot_confusion_matrix(self, save_path: str = 'confusion_matrix.png',  
                            figsize: Tuple[int, int] = (10, 8)): 
        """Plot and save confusion matrix""" 
        cm = self.get_confusion_matrix() 
        
        plt.figure(figsize=figsize) 
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names) 
        plt.title('Confusion Matrix') 
        plt.xlabel('Predicted') 
        plt.ylabel('Actual') 
        plt.tight_layout() 
        plt.savefig(save_path, dpi=300, bbox_inches='tight') 
        plt.close() 
        
        # Also save normalized version 
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
        plt.figure(figsize=figsize) 
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names) 
        plt.title('Normalized Confusion Matrix') 
        plt.xlabel('Predicted') 
        plt.ylabel('Actual') 
        plt.tight_layout() 
        plt.savefig(save_path.replace('.png', '_normalized.png'), dpi=300, bbox_inches='tight') 
        plt.close() 
    
    def plot_roc_curves(self, save_path: str = 'roc_curves.png', 
                       figsize: Tuple[int, int] = (12, 8)): 
        """Plot ROC curves for each class""" 
        if not self.all_probabilities: 
            logging.warning("No probabilities available for ROC curves") 
            return 
        
        y_true = np.array(self.all_targets) 
        y_prob = np.array(self.all_probabilities) 
        
        # Binarize the output 
        y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes))) 
        
        plt.figure(figsize=figsize) 
        
        # Plot ROC curve for each class 
        for i in range(self.num_classes): 
            if self.num_classes == 2 and i == 0: 
                continue  # Skip for binary classification 
                
            if self.num_classes > 2: 
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i]) 
            else: 
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1]) 
            
            roc_auc = roc_auc_score(y_true_bin[:, i] if self.num_classes > 2 else y_true,  
                                  y_prob[:, i] if self.num_classes > 2 else y_prob[:, 1]) 
            
            plt.plot(fpr, tpr, linewidth=2,  
                    label=f'{self.class_names[i]} (AUC = {roc_auc:.2f})') 
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2) 
        plt.xlim([0.0, 1.0]) 
        plt.ylim([0.0, 1.05]) 
        plt.xlabel('False Positive Rate') 
        plt.ylabel('True Positive Rate') 
        plt.title('ROC Curves') 
        plt.legend(loc="lower right") 
        plt.grid(True, alpha=0.3) 
        plt.tight_layout() 
        plt.savefig(save_path, dpi=300, bbox_inches='tight') 
        plt.close() 
    
    def generate_classification_report(self) -> str: 
        """Generate detailed classification report""" 
        y_true = np.array(self.all_targets) 
        y_pred = np.array(self.all_predictions) 
        
        return classification_report(y_true, y_pred,  
                                   target_names=self.class_names, 
                                   digits=4) 


class CentralizedEvaluator: 
    """Comprehensive evaluation for centralized models""" 
    
    def __init__(self, model: nn.Module, device: torch.device,  
                 class_names: Optional[List[str]] = None): 
        self.model = model 
        self.device = device 
        self.class_names = class_names 
        self.metrics_calculator = MetricsCalculator( 
            num_classes=len(class_names) if class_names else 5, 
            class_names=class_names 
        ) 
    
    def evaluate(self, data_loader: DataLoader, save_dir: str = "evaluation_results") -> Dict[str, float]: 
        """Comprehensive evaluation of the model""" 
        # Create save directory 
        os.makedirs(save_dir, exist_ok=True) 
        
        self.model.eval() 
        self.metrics_calculator.reset() 
        
        total_loss = 0.0 
        criterion = nn.CrossEntropyLoss() 
        
        logging.info("Starting evaluation...") 
        
        with torch.no_grad(): 
            for batch_idx, (data, targets) in enumerate(data_loader): 
                data, targets = data.to(self.device), targets.to(self.device) 
                
                outputs = self.model(data) 
                loss = criterion(outputs, targets) 
                total_loss += loss.item() 
                
                # Get predictions and probabilities 
                probabilities = torch.softmax(outputs, dim=1) 
                _, predicted = outputs.max(1) 
                
                # Update metrics 
                self.metrics_calculator.update(predicted, targets, probabilities) 
        
        # Compute all metrics 
        metrics = self.metrics_calculator.compute_metrics() 
        metrics['loss'] = total_loss / len(data_loader) 
        
        # Generate and save visualizations 
        self.metrics_calculator.plot_confusion_matrix( 
            os.path.join(save_dir, 'confusion_matrix.png') 
        ) 
        self.metrics_calculator.plot_roc_curves( 
            os.path.join(save_dir, 'roc_curves.png') 
        ) 
        
        # Generate classification report 
        report = self.metrics_calculator.generate_classification_report() 
        
        # Save results 
        self._save_results(metrics, report, save_dir) 
        
        logging.info("Evaluation completed") 
        return metrics 
    
    def _save_results(self, metrics: Dict[str, float], report: str, save_dir: str): 
        """Save evaluation results""" 
        # Save metrics as JSON 
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f: 
            # Convert numpy types to Python types for JSON serialization 
            serializable_metrics = {} 
            for k, v in metrics.items(): 
                if isinstance(v, (np.float64, np.float32)): 
                    serializable_metrics[k] = float(v) 
                elif isinstance(v, (np.int64, np.int32)): 
                    serializable_metrics[k] = int(v) 
                else: 
                    serializable_metrics[k] = v 
            json.dump(serializable_metrics, f, indent=2) 
        
        # Save classification report 
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f: 
            f.write(report) 
        
        # Log key metrics 
        logging.info("=" * 50) 
        logging.info("EVALUATION RESULTS") 
        logging.info("=" * 50) 
        logging.info(f"Loss: {metrics['loss']:.4f}") 
        logging.info(f"Accuracy: {metrics['accuracy']:.4f}") 
        logging.info(f"F1 Score (Macro): {metrics['f1_macro']:.4f}") 
        logging.info(f"Precision (Macro): {metrics['precision_macro']:.4f}") 
        logging.info(f"Recall (Macro): {metrics['recall_macro']:.4f}") 
        if 'roc_auc_macro' in metrics: 
            logging.info(f"ROC AUC (Macro): {metrics['roc_auc_macro']:.4f}") 
        logging.info("=" * 50) 


def count_parameters(model: nn.Module) -> int: 
    """Count the number of trainable parameters in a model""" 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)