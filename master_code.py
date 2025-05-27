import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from transforms_pipeline import train_transform, test_transform
from dataset_pipeline import CustomMelanomaDataset
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import timm
import os
import torch.backends.mps
import random
import numpy as np
import time
# Added for evaluation and plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, classification_report
from sklearn.preprocessing import label_binarize
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import json
import csv

# Global constants
NUM_EPOCHS = 30

def create_optimizer(model, lr_early_layers=5e-6, lr_later_layers=1e-4, lr_classifier=1e-3, weight_decay=2e-4):
    """Create optimizers with different learning rates for different parts of the model."""
    optimizer = torch.optim.AdamW(
        [
            {'params': [p for i, layer in enumerate(model.features) for p in layer.parameters() if i < 14], 'lr': lr_early_layers},
            {'params': [p for i, layer in enumerate(model.features) for p in layer.parameters() if i >= 14], 'lr': lr_later_layers},
            {'params': model.classifier.parameters(), 'lr': lr_classifier}
        ],
        weight_decay=weight_decay
    )
    return optimizer

def calculate_class_weights(train_dataset):
    """Calculate class weights to handle imbalanced datasets."""
    class_counts = np.bincount(train_dataset.labels)
    total_samples = len(train_dataset.labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    return torch.FloatTensor(class_weights)

def numpy_to_python(obj):
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(i) for i in obj]
    else:
        return obj

def plot_training_history(train_losses, test_losses, train_accuracies, test_accuracies, save_dir, batch_acc_per_epoch=None, batch_loss_per_epoch=None):
    """
    Plot accuracy/loss vs epoch using the provided lists (train/test losses and accuracies).
    Optionally overlays batch-wise accuracy/loss for more detailed visualization.
    Plots are saved to save_dir as accuracy_vs_epoch.png and loss_vs_epoch.png.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    acc_path = os.path.join(save_dir, 'accuracy_vs_epoch.png')
    loss_path = os.path.join(save_dir, 'loss_vs_epoch.png')

    def is_valid_list(lst):
        return (isinstance(lst, (list, np.ndarray)) and len(lst) > 0 and all(x is not None and not (isinstance(x, float) and np.isnan(x)) for x in lst))
    lengths = list(map(len, [train_losses, test_losses, train_accuracies, test_accuracies]))
    if not all(lengths):
        print("[ERROR] One or more input lists are invalid or empty. Skipping plots.")
        return None, None
    if len(set(lengths)) != 1:
        print(f"[ERROR] Input lists have mismatched lengths: {lengths}. Skipping plots.")
        return None, None
    epochs = list(range(1, lengths[0] + 1))

    # Accuracy vs Epoch plot
    plt.figure()
    plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o', color='tab:blue')
    plt.plot(epochs, test_accuracies, label='Test Accuracy', marker='o', color='tab:orange')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Epoch')
    plt.legend()
    plt.xticks(epochs)
    plt.tight_layout()
    plt.savefig(acc_path)
    plt.close()

    # Loss vs Epoch plot
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss', marker='o', color='tab:blue')
    plt.plot(epochs, test_losses, label='Test Loss', marker='o', color='tab:orange')
    # Overlay batch-wise loss if provided

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch (with batch-wise)')
    plt.legend()
    plt.xticks(epochs)
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

    print(f"[CLEAN] Plots saved to {acc_path} and {loss_path} using direct lists.")
    return acc_path, loss_path


# For reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Directory for model metrics
METRICS_BASE_DIR = '/Users/harshitkapoor/Downloads/model_metrics'
os.makedirs(METRICS_BASE_DIR, exist_ok=True)

# ResNet50
def setup_model_resnet50(num_classes):
    """Setup ResNet50 model with improved classifier."""
    model = models.resnet50(weights='IMAGENET1K_V2')
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze the last layer (layer4)
    for param in model.layer4.parameters():
        param.requires_grad = True
    # Modify the final fully connected layer with dropout
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model

# MobileNetV2
def setup_model_mobilenetv2(num_classes=7):
    """Setup MobileNetV2 model with improved architecture"""
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    # Freeze early layers
    for param in model.features[:14].parameters():
        param.requires_grad = False
    # Replace classifier with architecture matching the checkpoint
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),  # Updated dropout rate based on findings
        nn.Linear(model.last_channel, num_classes)
    )
    return model

# Xception
def setup_model_xception(num_classes=7, dropout=0.3463044926064332):
    """Setup Xception model with improved architecture and tunable dropout"""
    # Load Xception from timm
    model = timm.create_model('xception', pretrained=True)
    
    # Freeze early layers
    for name, param in model.named_parameters():
        if 'block1' in name or 'block2' in name:
            param.requires_grad = False
    
    # Replace classifier with improved architecture
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(512, num_classes)
    )
    
    return model

# VGG19
def setup_model_vgg19(num_classes):
    model = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)
    # Freeze all convolutional layers
    for param in model.features.parameters():
        param.requires_grad = False
    # Unfreeze last block (block 5)
    for param in list(model.features.parameters())[-10:]:
        param.requires_grad = True
    # Replace classifier with dropout and custom output layer
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, num_classes)
    )
    return model

# DenseNet
def setup_model_densenet(num_classes):
    # Load pre-trained DenseNet model
    model = models.densenet121(weights="IMAGENET1K_V1")
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze last denseblock and classifier for transfer learning
    for param in model.features.denseblock4.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, num_classes)
    )
    return model


# EfficientNet
# def setup_model_efficientnet(num_classes):
#     # Load pre-trained EfficientNet model
#     model = timm.create_model('tf_efficientnet_b0', pretrained=True)
#     # Freeze all layers
#     for param in model.parameters():
#         param.requires_grad = False
#     # Unfreeze last block and classifier for transfer learning
#     for param in model.classifier.parameters():
#         param.requires_grad = True
#     num_ftrs = model.classifier.in_features
#     model.classifier = nn.Sequential(
#         nn.Dropout(0.2),
#         nn.Linear(num_ftrs, num_classes)
#     )
#     return model
def setup_model_efficientnet(num_classes):
    """Setup EfficientNet model with improved architecture"""
    # Create EfficientNet model with pre-trained weights
    model = timm.create_model('tf_efficientnet_b0', pretrained=True)
    
    # Freeze early layers
    for name, param in model.named_parameters():
        if 'blocks.0' in name or 'blocks.1' in name or 'blocks.2' in name:
            param.requires_grad = False
    
    # Replace classifier with improved architecture
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.4),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.4),
        nn.Linear(128, num_classes)
    )
    
    return model

def plot_batch_metrics(batch_losses, batch_accuracies, save_dir, epoch):
    import matplotlib.pyplot as plt
    import os
    plt.figure()
    plt.plot(batch_losses, label='Batch Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title(f'Batch-wise Loss (Epoch {epoch+1})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'batch_loss_epoch_{epoch+1}.png'))
    plt.close()

    plt.figure()
    plt.plot(batch_accuracies, label='Batch Accuracy')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    plt.title(f'Batch-wise Accuracy (Epoch {epoch+1})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'batch_accuracy_epoch_{epoch+1}.png'))
    plt.close()

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS, save_path='model_checkpoints'):
    import os, time, json
    import numpy as np
    from torch.cuda.amp import autocast, GradScaler
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
    from sklearn.preprocessing import label_binarize
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    scaler = GradScaler()
    best_acc = 0.0
    os.makedirs(save_path, exist_ok=True)
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    batch_accuracies_per_epoch = []
    batch_losses_per_epoch = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_times = []
        epoch_start_time = time.time()
        all_preds = []
        all_labels = []
        batch_losses = []
        batch_accuracies = []
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            batch_start = time.time()
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # Mixed precision training
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            batch_end = time.time()
            batch_times.append(batch_end - batch_start)
            batch_losses.append(loss.item())
            batch_acc = (predicted == labels).float().mean().item()
            batch_accuracies.append(batch_acc)
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        batch_losses_per_epoch.append(batch_losses)
        batch_accuracies_per_epoch.append(batch_accuracies)
        # Plot batch-wise metrics for this epoch
        plot_batch_metrics(batch_losses, batch_accuracies, save_path, epoch)
        
        train_acc = 100. * correct / total
        train_losses.append(running_loss/len(train_loader))
        train_accs.append(train_acc)
        epoch_time = time.time() - epoch_start_time
        
        # Calculate additional train metrics
        train_metrics = {
            'loss': running_loss/len(train_loader),
            'accuracy': train_acc,
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
            'epoch_time': epoch_time,
            'avg_batch_time': sum(batch_times) / len(batch_times)
        }
        
        # Save epoch train metrics as JSON
        with open(os.path.join(save_path, f'train_metrics_epoch_{epoch+1}.json'), 'w') as f:
            json.dump(numpy_to_python(train_metrics), f, indent=4)
        
        # Test phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_all_preds = []
        val_all_labels = []
        val_all_probs = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                val_all_preds.extend(predicted.cpu().numpy())
                val_all_labels.extend(labels.cpu().numpy())
                probs = torch.softmax(outputs, dim=1)
                val_all_probs.extend(probs.cpu().numpy())
        
        val_acc = 100. * val_correct / val_total
        val_losses.append(val_loss/len(test_loader))
        val_accs.append(val_acc)
        
        # Calculate additional test metrics
        try:
            val_precision = precision_score(val_all_labels, val_all_preds, average='weighted', zero_division=0)
            val_recall = recall_score(val_all_labels, val_all_preds, average='weighted', zero_division=0)
            val_f1 = f1_score(val_all_labels, val_all_preds, average='weighted', zero_division=0)
            val_cm = confusion_matrix(val_all_labels, val_all_preds)
            
            # ROC AUC (with error handling)
            y_true_onehot = label_binarize(val_all_labels, classes=list(range(len(set(val_all_labels)))))
            val_roc_auc = None
            if len(set(val_all_labels)) > 1:
                try:
                    val_roc_auc = roc_auc_score(y_true_onehot, np.array(val_all_probs), multi_class='ovr', average='weighted')
                except Exception as e:
                    print(f"Could not calculate ROC AUC: {e}")
            
            val_report = classification_report(val_all_labels, val_all_preds, output_dict=True, zero_division=0)
        except Exception as e:
            val_precision = val_recall = val_f1 = val_roc_auc = None
            val_report = {}
            val_cm = None
        
        test_metrics = {
            'loss': val_loss/len(test_loader),
            'accuracy': val_acc,
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1,
            'roc_auc': val_roc_auc,
            'per_class_metrics': val_report,
            'confusion_matrix': val_cm.tolist() if val_cm is not None else None
        }
        
        with open(os.path.join(save_path, f'test_metrics_epoch_{epoch+1}.json'), 'w') as f:
            json.dump(numpy_to_python(test_metrics), f, indent=4)
        
        # Log training progress
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {running_loss/len(train_loader):.4f}')
        print(f'Training Accuracy: {train_acc:.2f}%')
        print(f'Test Loss: {val_loss/len(test_loader):.4f}')
        print(f'Test Accuracy: {val_acc:.2f}%')
        
        # Update scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_acc)
        else:
            scheduler.step()
        
        # Save model at specific epochs and if it's the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }, os.path.join(save_path, 'best_model.pth'))
        
        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy': val_acc,
            }, os.path.join(save_path, f'model_epoch_{epoch+1}.pth'))
    
    # Save last model
    torch.save({
        'epoch': num_epochs-1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy': val_accs[-1] if val_accs else None,
    }, os.path.join(save_path, 'last_model.pth'))
    
    # Save training metrics as JSON
    training_metrics = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    }
    
    # Save metrics as JSON (robust serialization)
    with open(os.path.join(save_path, 'metrics_summary.json'), 'w') as f:
        json.dump(numpy_to_python(training_metrics), f, indent=4)
    
    # Create and save training curves plots
    plot_training_history(
        train_losses, val_losses, train_accs, val_accs, save_path,
        batch_acc_per_epoch=batch_accuracies_per_epoch,
        batch_loss_per_epoch=batch_losses_per_epoch
    )
    
    return model, train_losses, val_losses, train_accs, val_accs, batch_accuracies_per_epoch, batch_losses_per_epoch

def validate(model, dataloader, loss_fn, device):
    """Validate model on testing set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if device == "mps":
                images = images.float()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return val_loss, val_acc, f1, all_preds, all_labels

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set and return comprehensive metrics."""
    import numpy as np
    from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, classification_report, roc_auc_score, roc_curve
    from sklearn.preprocessing import label_binarize
    
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    val_loss, val_acc, f1, all_preds, all_labels = validate(model, test_loader, loss_fn, device)
    
    all_probs = []
    with torch.no_grad():
        for images, labels in test_loader:
            if device == "mps":
                images = images.float()
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
    
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
            train_y_true_onehot = label_binarize(all_labels, classes=list(range(len(set(all_labels)))))
            # No softmax outputs for train, so set to NaN
            roc_auc = roc_auc_score(y_true_bin, all_probs, average='weighted', multi_class='ovr')
    except Exception:
        roc_auc = None
    
    fpr = dict()
    tpr = dict()
    roc_auc_per_class = dict()
    
    for i in range(num_classes):
        try:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
            roc_auc_per_class[i] = roc_auc_score(y_true_bin[:, i], all_probs[:, i])
        except Exception:
            fpr[i], tpr[i], roc_auc_per_class[i] = [], [], None
    
    # Classification report
    if not hasattr(test_loader.dataset, 'idx_to_class'):
        test_loader.dataset.idx_to_class = {idx: cls_name for cls_name, idx in test_loader.dataset.class_to_idx.items()}
    
    target_names = [test_loader.dataset.idx_to_class[i] for i in range(len(test_loader.dataset.class_to_idx))]
    report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True, zero_division=0)
    
    metrics = {
        'accuracy': val_acc / 100.0,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f1_score': f1,  # Ensure compatibility with save_model
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

def plot_confusion_matrix(cm, class_names, save_dir):
    """Plot confusion matrix and save to save_dir."""
    import os
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    return cm_path

def save_evaluation_log(metrics, save_dir, cm_path=None, roc_path=None, acc_path=None, loss_path=None):
    import os
    log_path = os.path.join(save_dir, 'evaluation_log.txt')
    with open(log_path, 'w') as logf:
        for k, v in metrics.items():
            if k not in ['fpr', 'tpr', 'predictions', 'true_labels']:  # Skip large arrays
                logf.write(f"{k}: {v}\n")
        if cm_path:
            logf.write(f"Confusion matrix plot: {cm_path}\n")
        if roc_path:
            logf.write(f"ROC curve plot: {roc_path}\n")
        if acc_path and os.path.exists(acc_path):
            logf.write(f"Accuracy vs Epoch plot saved as: {acc_path}\n")
        if loss_path and os.path.exists(loss_path):
            logf.write(f"Loss vs Epoch plot saved as: {loss_path}\n")

def save_model(model, optimizer, metrics, class_mapping, filename):
    """Save model with all relevant information."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': metrics['accuracy'],
        'f1_score': metrics['f1_score'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'class_mapping': class_mapping
    }, filename)
    print(f"\nModel saved to {filename}")

def main():
    print("[MONITOR] Starting main()")
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using {device.type} device")

    # Dataset paths
    base_dir = os.path.expanduser("~/Downloads/Split_Dataset")
    train_img_dir = os.path.join(base_dir, "train")
    test_img_dir = os.path.join(base_dir, "test")
    if not os.path.exists(train_img_dir) or not os.path.exists(test_img_dir):
        raise FileNotFoundError(f"Dataset directories not found")

    # Create datasets
    train_dataset = CustomMelanomaDataset(img_dir=train_img_dir, transform=train_transform)
    test_dataset = CustomMelanomaDataset(img_dir=test_img_dir, transform=test_transform)
    num_classes = len(train_dataset.classes)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    print(f"[MONITOR] Train loader length: {len(train_loader)}")
    print(f"[MONITOR] Test loader length: {len(test_loader)}")

    # Model configs: setup, optimizer, scheduler per model
    MODEL_CONFIGS = {
            # "resnet50": {
            #     "setup": lambda: setup_model_resnet50(num_classes=num_classes),
            #     "optimizer": lambda model: torch.optim.AdamW([
            #         {'params': [p for n, p in model.named_parameters() if 'layer4' in n], 'lr': 5e-5},
            #         {'params': model.fc.parameters(), 'lr': 1e-3}
            #     ], weight_decay=1e-4),
            #     "scheduler": lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(
            #         optimizer, mode='max', factor=0.1, patience=4, verbose=True)
            # },
            # "mobilenetv2": {
            #     "setup": lambda: setup_model_mobilenetv2(num_classes=num_classes),
            #     "optimizer": lambda model: torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4),
            #     "scheduler": lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(
            #         optimizer, mode='max', factor=0.1, patience=5, verbose=True)
            # },
            # "xception": {
            #     "setup": lambda: setup_model_xception(num_classes=num_classes, dropout=0.5),
            #     "optimizer": lambda model: torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4),
            #     "scheduler": lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(
            #         optimizer, mode='max', factor=0.1, patience=5, verbose=True)
            # },
            "vgg19": {
                "setup": lambda: setup_model_vgg19(num_classes=num_classes),
                "optimizer": lambda model: torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4),
                "scheduler": lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            },
            "densenet": {
                "setup": lambda: setup_model_densenet(num_classes=num_classes),
                "optimizer": lambda model: torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=0.01),
                "scheduler": lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='max',
                    factor=0.1,
                    patience=5,
                    verbose=True
                ),
            },
            "efficientnet": {
                "setup": lambda: setup_model_efficientnet(num_classes=num_classes),
                "optimizer": lambda model: torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4),
                "scheduler": lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='max',
                    factor=0.1,
                    patience=5,
                    verbose=True
                ),
            }
        }

    def calculate_class_weights(train_dataset):
        """Calculate class weights to handle imbalanced datasets."""
        class_counts = np.bincount(train_dataset.labels)
        total_samples = len(train_dataset.labels)
        class_weights = total_samples / (len(class_counts) * class_counts)
        return torch.FloatTensor(class_weights)

    class_weights = calculate_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for model_name, config in MODEL_CONFIGS.items():
        print(f"\n========== Training {model_name.upper()} ==========")
        metrics_dir = os.path.join(METRICS_BASE_DIR, model_name)
        os.makedirs(metrics_dir, exist_ok=True)
        model = config["setup"]()
        optimizer = config["optimizer"](model)
        scheduler = config["scheduler"](optimizer)
        trained_model, train_losses, val_losses, train_accs, val_accs, batch_acc_per_epoch, batch_loss_per_epoch = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=NUM_EPOCHS,
            save_path=metrics_dir
        )
        print(f"Training completed for {model_name}!")
        print(f"[MONITOR] Evaluating {model_name.upper()}...")
        metrics = evaluate_model(model, test_loader, device)
        print(f"[MONITOR] Evaluation completed for {model_name.upper()}")
        class_names = list(train_dataset.idx_to_class.values())
        cm = np.array(metrics['confusion_matrix'])
        import json, csv

        with open(os.path.join(metrics_dir, 'test_metrics.json'), 'w') as f:
            json.dump(numpy_to_python(metrics), f, indent=4)
        report = metrics.get('classification_report', {})
        with open(os.path.join(metrics_dir, 'classification_report.csv'), 'w', newline='') as csvfile:
            fieldnames = ['class', 'precision', 'recall', 'f1-score', 'support']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for idx, cname in enumerate(class_names):
                metrics_row = report.get(cname, None)
                if metrics_row:
                    writer.writerow({
                        'class': cname,
                        'precision': metrics_row['precision'],
                        'recall': metrics_row['recall'],
                        'f1-score': metrics_row['f1-score'],
                        'support': metrics_row['support']
                    })
            for avg_type in ['macro avg', 'weighted avg']:
                if avg_type in report:
                    metrics_row = report[avg_type]
                    writer.writerow({
                        'class': avg_type,
                        'precision': metrics_row['precision'],
                        'recall': metrics_row['recall'],
                        'f1-score': metrics_row['f1-score'],
                        'support': metrics_row['support']
                    })

        print(f"[MONITOR] Saving metrics for {model_name.upper()}...")
        # # Print summary metrics
        # print("\n--- Evaluation on Test Set ---")
        # print(f"Accuracy: {metrics['accuracy']:.4f}")
        # print(f"Precision: {metrics['precision']:.4f}")
        # print(f"Recall: {metrics['recall']:.4f}")
        # print(f"F1 Score: {metrics['f1']:.4f}")
        # print(f"Sensitivity (Recall): {metrics['sensitivity']:.4f}")
        # print(f"Specificity: {metrics['specificity']:.4f}")
        # print(f"AUC (weighted): {metrics['roc_auc']:.4f}")
        # print("Confusion Matrix:\n", cm)

        # # Save confusion matrix plot
        cm_path = plot_confusion_matrix(cm, class_names, metrics_dir)
        roc_path = None
        if 'fpr' in metrics and 'tpr' in metrics and 'roc_auc_per_class' in metrics:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 6))
            for i in range(len(class_names)):
                if len(metrics['fpr'][i]) > 0 and len(metrics['tpr'][i]) > 0:
                    plt.plot(metrics['fpr'][i], metrics['tpr'][i], label=f'{class_names[i]} (AUC = {metrics["roc_auc_per_class"][i]:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.tight_layout()
            roc_path = os.path.join(metrics_dir, 'roc_curve.png')
            plt.savefig(roc_path)
            plt.close()
            print(f"ROC curve saved as: {roc_path}")
        acc_path, loss_path = plot_training_history(
            train_losses, val_losses, train_accs, val_accs, metrics_dir,
            batch_acc_per_epoch=batch_acc_per_epoch,
            batch_loss_per_epoch=batch_loss_per_epoch
        )
        print(f"[MONITOR] Saving evaluation log for {model_name.upper()}...")
        save_evaluation_log(metrics, metrics_dir, cm_path=cm_path, roc_path=roc_path, acc_path=acc_path, loss_path=loss_path)
        print(f"[MONITOR] Evaluation log saved for {model_name.upper()}")
        print(f"[MONITOR] Saving model for {model_name.upper()}...")
        save_model(model, optimizer, metrics, train_dataset.class_to_idx, os.path.join(metrics_dir, "improved.pth"))
        print(f"[MONITOR] Model saved for {model_name.upper()}")
        print(f"Results saved in {metrics_dir}")
        print(f"[MONITOR] Metrics saved for {model_name.upper()}")
    print("[MONITOR] Main function completed")

if __name__ == "__main__":
    main()
