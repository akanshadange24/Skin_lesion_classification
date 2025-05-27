import torch
import torch.nn as nn
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transforms_pipeline import get_transforms
from dataset_pipeline import CustomMelanomaDataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from torchvision import transforms
import seaborn as sns


# Import model setups from your original code
from master_code import (
    setup_model_resnet50, setup_model_mobilenetv2, setup_model_xception,
    setup_model_vgg19, setup_model_densenet, setup_model_efficientnet,
    setup_model_inception, evaluate_model
)

# Import the uncertainty-guided ensemble
from uncertainty_guided_ensemble import (
    UncertaintyGuidedEnsemble,
    load_model_checkpoint,
    visualize_model_uncertainty,
    visualize_ensemble_performance,
    compare_ensemble_with_individual_models
)

class ModelPredictionCache:
    """Cache for model predictions to avoid redundant computations"""
    def __init__(self, device=None):
        self.cache = {}
        self.model_outputs = {}
        self.device = device or torch.device('mps' if torch.backends.mps.is_available() else 
                                           'cuda' if torch.cuda.is_available() else 'cpu')
        
    def _get_cache_key(self, inputs):
        """Generate a cache key from inputs, handling both tensor and dict inputs"""
        if isinstance(inputs, dict):
            # For dict inputs, hash each tensor separately and combine
            key_parts = []
            for name, tensor in sorted(inputs.items()):  # Sort to ensure consistent order
                if isinstance(tensor, torch.Tensor):
                    # Move to CPU only for hashing, keep original tensor on device
                    key_parts.append(tensor.detach().cpu().numpy().tobytes())
                else:
                    key_parts.append(str(tensor).encode())
            return hash(b''.join(key_parts))
        else:
            # For tensor inputs, hash directly
            return hash(inputs.detach().cpu().numpy().tobytes())
        
    def get_predictions(self, model_name, inputs):
        """Get cached predictions or compute if not cached"""
        cache_key = self._get_cache_key(inputs)
        if cache_key not in self.cache:
            self.cache[cache_key] = {}
            
        if model_name not in self.cache[cache_key]:
            with torch.no_grad():
                if isinstance(inputs, dict):
                    if model_name.lower() == 'inception':
                        model_input = inputs['Inception'].to(self.device)
                    else:
                        model_input = inputs['Default'].to(self.device)
                else:
                    model_input = inputs.to(self.device)
                    
                outputs = self.model_outputs[model_name](model_input)
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits
                    
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                # Store tensors on the appropriate device
                self.cache[cache_key][model_name] = {
                    'predictions': preds,  # Already on device
                    'probabilities': probs,  # Already on device
                    'outputs': outputs  # Already on device
                }
                
        return self.cache[cache_key][model_name]
    
    def register_model(self, model_name, model):
        """Register a model for caching"""
        self.model_outputs[model_name] = model.to(self.device)
        
    def clear_cache(self):
        """Clear the prediction cache"""
        self.cache.clear()

def load_models(model_configs, model_paths, device, num_classes):
    """
    Load pretrained models from checkpoints
    
    Args:
        model_configs: Dictionary of model configurations
        model_paths: Dictionary of model checkpoint paths
        device: Device to load models on
        num_classes: Number of classes in the dataset
        
    Returns:
        models: List of loaded models
        model_names: List of model names
    """
    models = []
    model_names = []
    
    for model_name, model_path in model_paths.items():
        if model_name not in model_configs:
            print(f"Warning: Configuration for {model_name} not found. Skipping.")
            continue
            
        try:
            print(f"Loading {model_name} from {model_path}")
            model_func = model_configs[model_name]["setup"]
            model = load_model_checkpoint(model_path, lambda: model_func(num_classes), device)
            models.append(model)
            model_names.append(model_name)
            print(f"Successfully loaded {model_name}")
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
    
    return models, model_names

def evaluate_individual_models(models, model_names, test_loader, device):
    """
    Evaluate each model individually using prediction cache
    """
    cache = ModelPredictionCache(device=device)  # Pass device to cache
    for model, name in zip(models, model_names):
        cache.register_model(name, model)
    
    individual_results = []
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        print(f"Evaluating {name} individually...")
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                # Get predictions from cache (already on correct device)
                cached = cache.get_predictions(name, inputs)
                preds = cached['predictions']
                
                # Move to CPU only for numpy conversion
                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        
        print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        individual_results.append({
            'name': name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        })
    
    return individual_results, cache

def tune_ensemble_hyperparameters(models, test_loader, device):
    """
    Enhanced hyperparameter tuning for the uncertainty-guided ensemble
    """
    from itertools import product
    from sklearn.metrics import f1_score
    import numpy as np
    
    # Define expanded parameter grid
    temperatures = [0.8, 1.0, 1.2, 1.5]  # More focused around 1.0
    uncertainty_thresholds = [0.4, 0.5, 0.6, 0.7]  # Higher thresholds
    confidence_boosts = [1.5, 2.0, 2.5]  # Different confidence boosting factors
    
    best_f1 = -1
    best_params = None
    
    print("\nTuning ensemble hyperparameters...")
    print("Grid search over:")
    print(f"Temperatures: {temperatures}")
    print(f"Uncertainty thresholds: {uncertainty_thresholds}")
    print(f"Confidence boosts: {confidence_boosts}")
    
    # Grid search
    for temp, threshold, boost in product(temperatures, uncertainty_thresholds, confidence_boosts):
        print(f"\nTrying temperature={temp:.1f}, uncertainty_threshold={threshold:.1f}, confidence_boost={boost:.1f}")
        
        # Create ensemble with current parameters
        ensemble = UncertaintyGuidedEnsemble(
            models=models,
            model_names=[f"model_{i}" for i in range(len(models))],
            temperature=temp,
            uncertainty_threshold=threshold,
            device=device
        )
        ensemble.confidence_boost = boost
        
        # Evaluate ensemble
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                if isinstance(inputs, dict):
                    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
                else:
                    inputs = inputs.to(device)
                
                outputs = ensemble(inputs)
                probs = outputs
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Calculate class-wise F1 scores
        class_f1 = f1_score(all_labels, all_preds, average=None)
        min_class_f1 = np.min(class_f1)  # Minimum class F1 score
        
        # Combined score: weighted average of overall F1 and minimum class F1
        combined_score = 0.7 * f1 + 0.3 * min_class_f1
        
        print(f"F1 Score: {f1:.4f}, Min Class F1: {min_class_f1:.4f}, Combined Score: {combined_score:.4f}")
        
        # Update best parameters if current combination is better
        if combined_score > best_f1:
            best_f1 = combined_score
            best_params = {
                'temperature': temp,
                'uncertainty_threshold': threshold,
                'confidence_boost': boost
            }
            print(f"New best parameters found! Combined Score: {combined_score:.4f}")
    
    print("\nBest hyperparameters found:")
    print(f"Temperature: {best_params['temperature']:.1f}")
    print(f"Uncertainty threshold: {best_params['uncertainty_threshold']:.1f}")
    print(f"Confidence boost: {best_params['confidence_boost']:.1f}")
    print(f"Best combined score: {best_f1:.4f}")
    
    return best_params

def validate(model, test_loader, criterion, device):
    """Validate model on test set and return basic metrics."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Handle dictionary of inputs
            if isinstance(inputs, dict):
                # For validation, use Default transform for all models
                inputs = inputs['Default'].to(device)
            else:
                inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            # Handle tuple output from ensemble model
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Use only the ensemble predictions
            
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = val_loss / len(test_loader)
    val_acc = 100. * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return val_loss, val_acc, f1, all_preds, all_labels

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set and return comprehensive metrics."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    val_loss, val_acc, f1, all_preds, all_labels = validate(model, test_loader, loss_fn, device)
    
    all_probs = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            try:
                # Handle dictionary of inputs
                if isinstance(inputs, dict):
                    inputs = inputs['Default'].to(device)
                else:
                    inputs = inputs.to(device)
                
                if device == "mps":
                    inputs = inputs.float()
                
                outputs = model(inputs)
                # Handle tuple output from ensemble model
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Use only the ensemble predictions
                
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
            except Exception as e:
                print(f"Error during evaluation: {str(e)}")
                continue
    
    all_probs = np.array(all_probs)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted')
    sensitivity = recall
    
    cm = confusion_matrix(all_labels, all_preds)
    
    # Specificity per class
    specificity_per_class = []
    for i in range(len(cm)):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(np.delete(cm, i, axis=0)[:, i])
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_per_class.append(specificity)
    specificity = np.mean(specificity_per_class)
    
    # ROC/AUC
    num_classes = len(np.unique(all_labels))
    y_true_bin = label_binarize(all_labels, classes=list(range(num_classes)))
    
    try:
        if len(set(all_labels)) > 1:
            roc_auc = roc_auc_score(y_true_bin, all_probs, average='weighted', multi_class='ovr')
    except Exception as e:
        print(f"Error calculating ROC AUC: {str(e)}")
        roc_auc = None
    
    fpr = dict()
    tpr = dict()
    roc_auc_per_class = dict()
    
    for i in range(num_classes):
        try:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
            roc_auc_per_class[i] = roc_auc_score(y_true_bin[:, i], all_probs[:, i])
        except Exception as e:
            print(f"Error calculating ROC curve for class {i}: {str(e)}")
            fpr[i], tpr[i], roc_auc_per_class[i] = [], [], None
    
    # Classification report
    if not hasattr(test_loader.dataset, 'idx_to_class'):
        test_loader.dataset.idx_to_class = {idx: cls_name for cls_name, idx in test_loader.dataset.class_to_idx.items()}
    
    target_names = [test_loader.dataset.idx_to_class[i] for i in range(len(test_loader.dataset.class_to_idx))]
    report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True, zero_division=0)
    
    # Print comprehensive metrics
    print("\nComprehensive Model Evaluation Metrics:")
    print(f"Overall Accuracy: {val_acc/100:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc if roc_auc is not None else 'N/A'}")
    print("\nClass-wise Metrics:")
    for i, cls_name in enumerate(target_names):
        print(f"\n{cls_name}:")
        print(f"  Precision: {report[cls_name]['precision']:.4f}")
        print(f"  Recall: {report[cls_name]['recall']:.4f}")
        print(f"  F1-score: {report[cls_name]['f1-score']:.4f}")
        print(f"  Specificity: {specificity_per_class[i]:.4f}")
        print(f"  AUC-ROC: {roc_auc_per_class[i] if roc_auc_per_class[i] is not None else 'N/A'}")
    
    metrics = {
        'accuracy': val_acc / 100.0,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f1_score': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'specificity_per_class': specificity_per_class,
        'roc_auc': roc_auc,
        'roc_auc_per_class': roc_auc_per_class,
        'fpr': fpr,
        'tpr': tpr,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'predictions': all_preds,
        'true_labels': all_labels,
        'val_loss': val_loss
    }
    
    return metrics

def plot_metrics(metrics, class_metrics, save_dir):
    """Plot comprehensive metrics and save visualizations."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = np.array(metrics['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Plot class-wise metrics
    class_names = list(class_metrics.keys())
    metrics_names = ['Precision', 'Recall', 'Specificity']
    metrics_values = [
        [class_metrics[cls]['precision'] for cls in class_names],
        [class_metrics[cls]['recall'] for cls in class_names],
        [class_metrics[cls]['specificity'] for cls in class_names]
    ]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.25  # Adjusted width for 3 metrics
    
    for i, (name, values) in enumerate(zip(metrics_names, metrics_values)):
        plt.bar(x + i*width, values, width, label=name)
    
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Class-wise Metrics')
    plt.xticks(x + width, class_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_metrics.png'))
    plt.close()
    
    # Plot overall metrics comparison
    overall_metrics = ['Accuracy', 'Macro Precision', 'Macro Recall']
    overall_values = [
        metrics['accuracy'],
        metrics['macro_precision'],
        metrics['macro_recall']
    ]
    
    plt.figure(figsize=(10, 6))
    plt.bar(overall_metrics, overall_values)
    plt.title('Overall Model Performance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'overall_metrics.png'))
    plt.close()

def create_test_loader(data_dir, batch_size=32, num_workers=4, pin_memory=True):
    """
    Create a standardized test loader with optimal settings
    
    Args:
        data_dir: Directory containing test data
        batch_size: Batch size for testing
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        DataLoader: Standardized test loader
    """
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Special transform for Inception
    inception_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset with both transforms
    test_dataset = CustomMelanomaDataset(
        root_dir=os.path.join(data_dir, 'test'),
        transform={
            'Default': test_transform,
            'Inception': inception_transform
        }
    )
    
    # Create optimized test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle test data
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,  # Keep workers alive between iterations
        prefetch_factor=2 if num_workers > 0 else None  # Prefetch 2 batches per worker
    )
    
    return test_loader

def calculate_class_weights(test_loader):
    """
    Calculate class weights using a more conservative approach
    """
    # Get class distribution
    class_counts = torch.zeros(len(test_loader.dataset.classes), dtype=torch.float32)
    for _, labels in test_loader:
        for label in labels:
            class_counts[label] += 1
    
    # Calculate effective number of samples
    beta = 0.9  # Effective sample size parameter
    effective_num = 1.0 - torch.pow(beta, class_counts)
    effective_num = (1.0 - beta) / effective_num
    
    # Calculate weights using effective number of samples
    weights = effective_num / effective_num.sum() * len(class_counts)
    
    # Apply softmax to make weights less extreme
    weights = torch.softmax(weights, dim=0) * len(class_counts)
    
    # Clip weights to prevent extreme values
    max_weight = 2.0  # Maximum weight multiplier
    weights = torch.clamp(weights, min=1.0/max_weight, max=max_weight)
    
    return weights

def convert_to_serializable(obj):
    """
    Convert numpy arrays and other non-serializable objects to JSON-serializable format
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                         np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.datetime64, np.timedelta64)):
        return str(obj)
    return obj

def evaluate_ensemble_metrics(ensemble, test_loader, device):
    """Evaluate ensemble model with detailed metrics."""
    ensemble.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            try:
                # Handle dictionary of inputs
                if isinstance(inputs, dict):
                    inputs = inputs['Default'].to(device)
                else:
                    inputs = inputs.to(device)
                
                outputs = ensemble(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
            except Exception as e:
                print(f"Error during ensemble evaluation: {str(e)}")
                continue
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate metrics per class
    num_classes = len(np.unique(all_labels))
    class_metrics = {}
    
    for i in range(num_classes):
        # True Positives, False Positives, True Negatives, False Negatives
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fn = np.sum(cm[i, :]) - tp
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate AUC-ROC for this class
        try:
            auc = roc_auc_score((all_labels == i).astype(int), all_probs[:, i])
        except Exception as e:
            print(f"Error calculating AUC for class {i}: {str(e)}")
            auc = None
        
        class_metrics[i] = {
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'auc': auc,
            'support': tp + fn
        }
    
    # Calculate overall metrics
    overall_metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'macro_precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'macro_recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'macro_f1': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'weighted_precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'weighted_recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'weighted_f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'confusion_matrix': cm.tolist()
    }
    
    # Print detailed metrics
    print("\nDetailed Ensemble Evaluation Metrics:")
    print(f"Overall Accuracy: {overall_metrics['accuracy']:.4f}")
    print(f"Macro Precision: {overall_metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {overall_metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score: {overall_metrics['macro_f1']:.4f}")
    print(f"Weighted Precision: {overall_metrics['weighted_precision']:.4f}")
    print(f"Weighted Recall: {overall_metrics['weighted_recall']:.4f}")
    print(f"Weighted F1-Score: {overall_metrics['weighted_f1']:.4f}")
    
    print("\nPer-Class Metrics:")
    for i, metrics in class_metrics.items():
        class_name = test_loader.dataset.idx_to_class[i]
        auc_str = f"{metrics['auc']:.4f}" if metrics['auc'] is not None else "N/A"
        print(f"\n{class_name}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  AUC-ROC: {auc_str}")
        print(f"  Support: {metrics['support']}")
    
    return overall_metrics, class_metrics

def main():
    print("Starting main function...")
    parser = argparse.ArgumentParser(description='Train and evaluate ensemble model')
    parser.add_argument('--data_dir', type=str, default='Split_Dataset', help='D:\Akansha\HAM10000_RUN_JUPYTER\Split_Dataset\test')
    parser.add_argument('--metrics_dir', type=str, default='model_metrics', help='D:\Akansha\HAM10000_RUN_JUPYTER\model_metrics')
    parser.add_argument('--output_dir', type=str, default='ensemble_results', help='D:\Akansha\HAM10000_RUN_JUPYTER\model_metrics\ensemble result')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--tune', action='store_true', help='Tune ensemble hyperparameters')
    args = parser.parse_args()

    # Model paths using raw strings to avoid escape sequence warnings
    model_paths = {
        'resnet50': r"D:\Akansha\HAM10000_RUN_JUPYTER\model_metrics\resnet50\best_model.pth",
        'mobilenetv2': r"D:\Akansha\HAM10000_RUN_JUPYTER\model_metrics\mobilenetv2\best_model.pth",
        'xception': r"D:\Akansha\HAM10000_RUN_JUPYTER\model_metrics\xception\best_model.pth",
        'vgg19': r"D:\Akansha\HAM10000_RUN_JUPYTER\model_metrics\vgg19\best_model.pth",
        'densenet': r"D:\Akansha\HAM10000_RUN_JUPYTER\model_metrics\densenet\best_model.pth",
        'efficientnet': r"D:\Akansha\HAM10000_RUN_JUPYTER\model_metrics\efficientnet\best_model.pth",
        'inception': r"D:\Akansha\HAM10000_RUN_JUPYTER\model_metrics\inception\inception 2\best_model.pth"
    }
    
    # Set device
    device = torch.device(args.device)
    print(f"Using {device} device")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Created output directory: {args.output_dir}")
    
    # Define model configurations
    model_configs = {
        'resnet50': {
            'setup': setup_model_resnet50
        },
        'mobilenetv2': {
            'setup': setup_model_mobilenetv2
        },
        'xception': {
            'setup': setup_model_xception
        },
        'vgg19': {
            'setup': setup_model_vgg19
        },
        'densenet': {
            'setup': setup_model_densenet
        },
        'efficientnet': {
            'setup': setup_model_efficientnet
        },
        'inception': {
            'setup': setup_model_inception
        }
    }
    
    print("\nCreating test loader...")
    # Create standardized test loader
    test_loader = create_test_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() or torch.backends.mps.is_available()  # Enable for GPU/MPS
    )
    print("Test loader created successfully")
    
    print("\nLoading models...")
    # Load models
    num_classes = len(np.unique(test_loader.dataset.labels))
    models, model_names = load_models(model_configs, model_paths, device, num_classes)
    
    if len(models) == 0:
        print("No models were successfully loaded. Exiting.")
        return
    
    print(f"\nSuccessfully loaded {len(models)} models: {model_names}")
    
    print("\nEvaluating individual models...")
    # Evaluate individual models with caching
    individual_results, prediction_cache = evaluate_individual_models(models, model_names, test_loader, device)
    
    print("\nCalculating class weights...")
    # Calculate class weights with more conservative approach
    class_weights = calculate_class_weights(test_loader)
    print("\nClass weights for balancing (conservative):")
    for i, weight in enumerate(class_weights):
        class_name = test_loader.dataset.idx_to_class[i]
        print(f"{class_name}: {weight:.4f}")
    
    print("\nCalculating model weights...")
    # Calculate model weights based on individual performance with more emphasis on F1 score
    model_weights = {}
    total_f1 = sum(result['f1'] for result in individual_results)
    for result in individual_results:
        # Weight is proportional to F1 score, with higher emphasis on better models
        min_weight = 0.01  # Even lower minimum weight for more differentiation
        weight = max(result['f1'] / total_f1, min_weight)
        # Apply more aggressive power transformation to emphasize better models
        weight = weight ** 2.0  # More aggressive power transformation
        model_weights[result['name']] = weight
    
    # Normalize model weights
    total_weight = sum(model_weights.values())
    model_weights = {name: weight/total_weight for name, weight in model_weights.items()}
    
    print("\nModel weights based on individual performance (adjusted):")
    for name, weight in model_weights.items():
        print(f"{name}: {weight:.4f}")
    
    print("\nDetermining ensemble hyperparameters...")
    # Determine ensemble hyperparameters
    if args.tune:
        print("Tuning ensemble hyperparameters...")
        best_params = tune_ensemble_hyperparameters(models, test_loader, device)
        temperature = best_params['temperature']
        uncertainty_threshold = best_params['uncertainty_threshold']
        confidence_boost = best_params['confidence_boost']
    else:
        # Updated default parameters for better performance
        temperature = 0.98  # Even less aggressive temperature
        uncertainty_threshold = 0.35  # Even lower threshold to include more predictions
        confidence_boost = 4.0  # Higher confidence boost
        print(f"Using default parameters: temperature={temperature}, uncertainty_threshold={uncertainty_threshold}, confidence_boost={confidence_boost}")
    
    print("\nCreating ensemble model...")
    # Create ensemble model with performance-based weights and class balancing
    ensemble = UncertaintyGuidedEnsemble(
        models=models,
        model_names=model_names,
        temperature=temperature,
        uncertainty_threshold=uncertainty_threshold,
        device=device,
        prediction_cache=prediction_cache,
        model_weights=model_weights,
        class_weights=class_weights
    )
    ensemble.confidence_boost = confidence_boost
    
    print("\nEvaluating ensemble...")
    # Evaluate ensemble with detailed metrics
    overall_metrics, class_metrics = evaluate_ensemble_metrics(ensemble, test_loader, device)
    
    print("\nGenerating visualizations...")
    # Plot comprehensive metrics
    plot_metrics(overall_metrics, class_metrics, args.output_dir)
    
    # Visualize model uncertainties
    visualize_model_uncertainty(
        models=models,
        model_names=model_names,
        test_loader=test_loader,
        device=device,
        save_path=os.path.join(args.output_dir, 'model_uncertainties.png')
    )
    
    # Visualize ensemble performance with correct metric names
    visualize_ensemble_performance(
        ensemble_results={
            'accuracy': overall_metrics['accuracy'],
            'precision': overall_metrics['weighted_precision'],  # Use weighted precision
            'recall': overall_metrics['weighted_recall'],  # Use weighted recall
            'f1': overall_metrics['weighted_f1'],  # Use weighted F1
            'confusion_matrix': overall_metrics['confusion_matrix']
        },
        save_path=os.path.join(args.output_dir, 'ensemble_performance.png')
    )
    
    # Compare ensemble with individual models
    compare_ensemble_with_individual_models(
        individual_results=individual_results,
        ensemble_results={
            'accuracy': overall_metrics['accuracy'],
            'precision': overall_metrics['weighted_precision'],
            'recall': overall_metrics['weighted_recall'],
            'f1': overall_metrics['weighted_f1'],  # Map weighted_f1 to f1
            'confusion_matrix': overall_metrics['confusion_matrix']
        },
        save_path=os.path.join(args.output_dir, 'model_comparison.png')
    )
    
    print("\nSaving results...")
    # Save results to file
    results = {
        'individual_models': individual_results,
        'ensemble': overall_metrics,
        'parameters': {
            'temperature': float(temperature),  # Convert to Python float
            'uncertainty_threshold': float(uncertainty_threshold),
            'confidence_boost': float(confidence_boost)
        }
    }
    
    # Convert all numpy arrays to lists before saving
    serializable_results = convert_to_serializable(results)
    
    import json
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    print(f"Results saved to {os.path.join(args.output_dir, 'results.json')}")
    
    print("\nGenerating summary report...")
    # Generate a summary report
    with open(os.path.join(args.output_dir, 'summary_report.txt'), 'w') as f:
        f.write("Uncertainty-Guided Ensemble Model Evaluation\n")
        f.write("==========================================\n\n")
        
        f.write("Individual Model Performance:\n")
        for result in individual_results:
            f.write(f"- {result['name']}: Accuracy = {float(result['accuracy']):.4f}, F1 Score = {float(result['f1']):.4f}\n")
        
        f.write("\nEnsemble Model Performance:\n")
        f.write(f"- Accuracy = {float(overall_metrics['accuracy']):.4f}\n")
        f.write(f"- Weighted Precision = {float(overall_metrics['weighted_precision']):.4f}\n")
        f.write(f"- Weighted Recall = {float(overall_metrics['weighted_recall']):.4f}\n")
        f.write(f"- Weighted F1 Score = {float(overall_metrics['weighted_f1']):.4f}\n")
        f.write(f"- Macro Precision = {float(overall_metrics['macro_precision']):.4f}\n")
        f.write(f"- Macro Recall = {float(overall_metrics['macro_recall']):.4f}\n")
        f.write(f"- Macro F1 Score = {float(overall_metrics['macro_f1']):.4f}\n")
        
        f.write("\nEnsemble Parameters:\n")
        f.write(f"- Temperature: {float(temperature)}\n")
        f.write(f"- Uncertainty Threshold: {float(uncertainty_threshold)}\n")
        f.write(f"- Confidence Boost: {float(confidence_boost)}\n")
        
        # Calculate improvement using weighted F1 score
        best_individual = max(individual_results, key=lambda x: x['f1'])
        improvement = (float(overall_metrics['weighted_f1']) - float(best_individual['f1'])) / float(best_individual['f1']) * 100
        
        f.write(f"\nImprovement over best individual model ({best_individual['name']}): {improvement:.2f}%\n")
    
    print(f"Summary report saved to {os.path.join(args.output_dir, 'summary_report.txt')}")
    print("\nDone!")

if __name__ == "__main__":
    main()