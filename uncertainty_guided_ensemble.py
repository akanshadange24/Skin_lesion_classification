import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn as nn

class UncertaintyGuidedEnsemble(nn.Module):
    """
    Uncertainty-Guided Dynamic Ensemble that weights models based on their prediction uncertainty
    """
    def __init__(self, models, model_names, temperature=1.0, uncertainty_threshold=0.5, device=None, prediction_cache=None, model_weights=None, class_weights=None):
        """
        Initialize the ensemble with class balancing
        
        Args:
            models: List of pretrained models
            model_names: List of model names corresponding to the models
            temperature: Temperature parameter for softmax smoothing
            uncertainty_threshold: Threshold for uncertainty-based model selection
            device: Device to run the ensemble on
            prediction_cache: Prediction cache object
            model_weights: Optional dictionary of base weights for each model
            class_weights: Optional tensor of class weights for balancing
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.model_names = model_names
        self.temperature = temperature
        self.uncertainty_threshold = uncertainty_threshold
        self.device = device
        self.prediction_cache = prediction_cache
        self.confidence_boost = 4.0  # Increased further for stronger confidence emphasis
        
        # Initialize model weights based on performance if provided
        if model_weights is not None:
            # Apply more aggressive power transformation to make weights more distinct
            weights = torch.tensor([model_weights[name] for name in model_names], 
                                 dtype=torch.float32, 
                                 device=device)
            self.base_weights = torch.pow(weights, 2.5)  # More aggressive power transformation
            self.base_weights = self.base_weights / self.base_weights.sum()
        else:
            self.base_weights = torch.ones(len(models), dtype=torch.float32, device=device) / len(models)
        
        # Initialize class weights if provided
        if class_weights is not None:
            weights = class_weights.clone().detach().to(device)
            # Apply power transformation but cap at 4.0x
            self.class_weights = torch.pow(weights, 0.5)  # More aggressive transformation
            self.class_weights = torch.clamp(self.class_weights, max=4.0)
            # Normalize to have a mean of 1.0
            self.class_weights = self.class_weights / self.class_weights.mean()
        else:
            self.class_weights = None
        
        # Initialize dynamic temperature scaling
        self.temperature_scaling = nn.Parameter(torch.ones(1, device=device) * 0.95)
    
    def forward(self, inputs):
        """
        Forward pass with conservative class balancing
        """
        all_probs = []
        all_uncertainties = []
        all_confidences = []
        all_outputs = []
        all_entropies = []  # Store individual entropies
        
        with torch.no_grad():
            for model, model_name in zip(self.models, self.model_names):
                if self.prediction_cache is not None:
                    cached = self.prediction_cache.get_predictions(model_name, inputs)
                    probs = cached['probabilities']
                    outputs = cached['outputs']
                else:
                    if isinstance(inputs, dict):
                        if 'inception' in model_name.lower():
                            model_input = inputs['Inception'].to(self.device)
                        else:
                            model_input = inputs['Default'].to(self.device)
                    else:
                        model_input = inputs.to(self.device)
                        
                    outputs = model(model_input)
                    if hasattr(outputs, 'logits'):
                        outputs = outputs.logits
                    probs = torch.softmax(outputs / self.temperature, dim=1)
                
                all_outputs.append(outputs)
                all_probs.append(probs)
                
                # Calculate prediction confidence with temperature scaling
                scaled_probs = torch.softmax(outputs / 0.2, dim=1)  # Even lower temperature for confidence
                confidence = torch.max(scaled_probs, dim=1)[0]
                
                # Calculate uncertainty using entropy with temperature scaling
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                uncertainty = entropy / torch.log(torch.tensor(probs.size(1), dtype=torch.float32))
                
                all_uncertainties.append(uncertainty)
                all_confidences.append(confidence)
                all_entropies.append(entropy)
        
        # Stack all tensors
        all_probs = torch.stack(all_probs, dim=0)  # [num_models, batch_size, num_classes]
        all_uncertainties = torch.stack(all_uncertainties, dim=0)  # [num_models, batch_size]
        all_confidences = torch.stack(all_confidences, dim=0)  # [num_models, batch_size]
        all_outputs = torch.stack(all_outputs, dim=0)  # [num_models, batch_size, num_classes]
        all_entropies = torch.stack(all_entropies, dim=0)  # [num_models, batch_size]
        
        # Calculate base weights from model performance
        base_weights = self.base_weights.view(-1, 1)  # [num_models, 1]
        
        # Calculate uncertainty weights with softer threshold
        uncertainty_weights = torch.exp(-all_uncertainties * 2.0)  # Increased sensitivity to uncertainty
        
        # Calculate confidence weights with higher boost
        confidence_weights = torch.pow(all_confidences, self.confidence_boost)
        
        # Calculate agreement weights (how much models agree with each other)
        mean_probs = torch.mean(all_probs, dim=0, keepdim=True)  # [1, batch_size, num_classes]
        agreement = 1.0 - torch.mean(torch.abs(all_probs - mean_probs), dim=2)  # [num_models, batch_size]
        agreement_weights = torch.pow(agreement, 3.0)  # More aggressive agreement weighting
        
        # Calculate diversity weights (reward models that disagree with the mean when they're confident)
        diversity = torch.abs(all_probs - mean_probs)  # [num_models, batch_size, num_classes]
        max_diversity = torch.max(diversity, dim=2)[0]  # [num_models, batch_size]
        diversity_weights = torch.pow(max_diversity * all_confidences, 2.0)  # Reward confident disagreements
        
        # Calculate entropy-based weights (reward models with lower entropy when confident)
        entropy_weights = torch.exp(-all_entropies) * all_confidences
        
        # Combine weights with more emphasis on confidence and agreement
        weights = base_weights * (
            0.15 * uncertainty_weights +    # Reduced uncertainty weight
            0.40 * confidence_weights +     # Increased confidence weight
            0.25 * agreement_weights +      # Agreement weight
            0.10 * diversity_weights +      # Added diversity weight
            0.10 * entropy_weights          # Added entropy weight
        )
        
        # Apply uncertainty threshold more softly
        uncertainty_factor = torch.exp(-(all_uncertainties - self.uncertainty_threshold).clamp(min=0) * 0.5)
        weights = weights * (0.9 + 0.1 * uncertainty_factor)  # Keep more weight for uncertain predictions
        
        # Normalize weights
        weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-10)
        
        # Weighted ensemble with temperature scaling
        weighted_probs = torch.sum(weights.unsqueeze(-1) * all_probs, dim=0)
        
        # Apply class weights if available
        if self.class_weights is not None:
            # Apply class weights more aggressively
            weighted_probs = weighted_probs * self.class_weights.view(1, -1)
            # Renormalize
            weighted_probs = weighted_probs / weighted_probs.sum(dim=1, keepdim=True)
        
        # Apply dynamic temperature scaling
        final_temperature = self.temperature_scaling.clamp(min=0.8, max=1.2)
        weighted_probs = torch.softmax(weighted_probs / final_temperature, dim=1)
        
        return weighted_probs

def load_model_checkpoint(checkpoint_path, model_setup_func, device):
    """
    Load a model from checkpoint
    
    Args:
        checkpoint_path: Path to the model checkpoint
        model_setup_func: Function to set up the model architecture
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    # Initialize model
    model = model_setup_func()
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model

def calculate_model_uncertainty(model, inputs, device):
    """
    Calculate prediction uncertainty for a model
    
    Args:
        model: The model
        inputs: Input data (tensor or dictionary of transformed inputs)
        device: Device to run calculation on
        
    Returns:
        Uncertainty scores
    """
    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    
    with torch.no_grad():
        # Handle dictionary of inputs
        if isinstance(inputs, dict):
            # For Inception models, use 'Inception' transform
            if 'Inception' in str(type(model).__name__):
                model_input = inputs['Inception'].to(device)
            else:
                model_input = inputs['Default'].to(device)
        else:
            model_input = inputs.to(device)
        
        outputs = model(model_input)
        
        # Handle different output formats
        if hasattr(outputs, 'logits') and hasattr(outputs, 'aux_logits'):
            logits = outputs.logits  # For InceptionV3
        else:
            logits = outputs
        
        probs = softmax(logits)
        
        # Calculate prediction entropy as uncertainty measure
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        uncertainty = entropy / np.log(probs.size(1))  # Normalize by log(num_classes)
    
    return uncertainty.cpu().numpy()

def visualize_model_uncertainty(models, model_names, test_loader, device, save_path):
    """
    Visualize uncertainty distribution for each model
    
    Args:
        models: List of models
        model_names: List of model names
        test_loader: DataLoader for test set
        device: Device to run visualization on
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(15, 10))
    
    all_uncertainties = []
    
    # Calculate uncertainties for all models
    for i, (model, name) in enumerate(zip(models, model_names)):
        model_uncertainties = []
        
        for inputs, _ in test_loader:
            uncertainty = calculate_model_uncertainty(model, inputs, device)
            model_uncertainties.extend(uncertainty)
        
        all_uncertainties.append(model_uncertainties)
    
    # Plot uncertainty distributions
    for i, (uncertainties, name) in enumerate(zip(all_uncertainties, model_names)):
        sns.kdeplot(uncertainties, label=name)
    
    plt.title('Uncertainty Distribution Across Models', fontsize=16)
    plt.xlabel('Uncertainty Score', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Model uncertainty visualization saved to {save_path}")

def visualize_ensemble_performance(ensemble_results, save_path):
    """
    Visualize ensemble performance metrics and confusion matrix
    
    Args:
        ensemble_results: Dictionary containing ensemble evaluation results
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    values = [ensemble_results[metric] for metric in metrics]
    
    axes[0].bar(metrics, values, color='steelblue')
    axes[0].set_title('Ensemble Performance Metrics', fontsize=16)
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3)
    
    for i, value in enumerate(values):
        axes[0].text(i, value + 0.02, f'{value:.3f}', ha='center', fontsize=12)
    
    # Plot confusion matrix
    cm = np.array(ensemble_results['confusion_matrix'])  # Convert list to numpy array
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=axes[1], cmap='Blues', colorbar=False)
    axes[1].set_title('Ensemble Confusion Matrix', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Ensemble performance visualization saved to {save_path}")

def compare_ensemble_with_individual_models(individual_results, ensemble_results, save_path):
    """
    Compare ensemble performance with individual models
    
    Args:
        individual_results: List of dictionaries containing individual model results
        ensemble_results: Dictionary containing ensemble evaluation results
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(14, 8))
    
    # Extract model names and F1 scores
    model_names = [result['name'] for result in individual_results]
    f1_scores = [result['f1'] for result in individual_results]
    
    # Add ensemble
    model_names.append('Ensemble')  # Use hardcoded name instead of trying to get from results
    f1_scores.append(ensemble_results['f1'])
    
    # Sort by F1 score
    sorted_indices = np.argsort(f1_scores)
    sorted_names = [model_names[i] for i in sorted_indices]
    sorted_scores = [f1_scores[i] for i in sorted_indices]
    
    # Plot F1 scores
    colors = ['steelblue'] * len(individual_results) + ['coral']
    bars = plt.barh(sorted_names, sorted_scores, color=colors)
    
    plt.title('F1 Score Comparison: Individual Models vs. Ensemble', fontsize=16)
    plt.xlabel('F1 Score', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, sorted_scores)):
        plt.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Model comparison visualization saved to {save_path}")