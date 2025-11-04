import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import timm  # For EfficientNet
from PIL import Image
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SpeedometerDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Dataset class for speedometer needle position detection
        
        Args:
            data_dir: Dictionary with 'stopped' and 'moving' folder paths
            transform: Image transformations
        """
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load images where train is stopped (label = 0)
        stopped_dir = data_dir['stopped']
        if os.path.exists(stopped_dir):
            for img_name in os.listdir(stopped_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(stopped_dir, img_name))
                    self.labels.append(0)  # 0 for stopped
        
        # Load images where train is moving (label = 1)
        moving_dir = data_dir['moving']
        if os.path.exists(moving_dir):
            for img_name in os.listdir(moving_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(moving_dir, img_name))
                    self.labels.append(1)  # 1 for moving
        
        print(f"Total images: {len(self.images)}")
        print(f"Stopped: {len(self.labels) - sum(self.labels)}, Moving: {sum(self.labels)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load and preprocess image
        try:
            image = Image.open(img_path)
            
            # Convert grayscale to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image in case of error
            image = torch.zeros(3, 224, 224)
        
        return image, label

# Enhanced data augmentation for speedometer images
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.1),  # Very minimal flip for speedometer
        transforms.RandomRotation(degrees=5),  # Small rotation as speedometer shouldn't rotate much
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.2),  # Handle grayscale variations
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),  # Handle blur from low quality
        # Add noise to simulate poor video quality
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

class EfficientNetSpeedometerDetector(nn.Module):
    def __init__(self, model_name='efficientnet_b0', num_classes=2, pretrained=True):
        super(EfficientNetSpeedometerDetector, self).__init__()
        
        # Load pre-trained EfficientNet
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        
        # Get the number of features from the classifier
        num_features = self.backbone.classifier.in_features
        
        # Replace classifier with custom head optimized for speedometer detection
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def train_model(model, train_loader, val_loader, num_epochs=25, learning_rate=0.001):
    """
    Training function optimized for speedometer needle detection
    """
    # Loss function with label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer with different learning rates for backbone and classifier
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': learning_rate * 0.1},  # Lower LR for pre-trained layers
        {'params': classifier_params, 'lr': learning_rate}
    ], weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training history
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stable training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        train_accuracy = 100 * train_correct / train_total
        avg_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_labels = []
        val_loss = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc='Validation')
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate additional metrics
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        # Update learning rate
        scheduler.step(val_accuracy)
        
        # Store metrics
        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        print('-' * 70)
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'model_name': 'efficientnet_b0',
                'num_classes': 2,
                'class_names': ['Stopped', 'Moving']
            }, 'best_speedometer_detection_model.pth')
            print(f'✓ New best model saved with validation accuracy: {val_accuracy:.2f}%')
    
    return train_losses, train_accuracies, val_accuracies

def plot_training_history(train_losses, train_accuracies, val_accuracies):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Train Accuracy', color='blue')
    ax2.plot(val_accuracies, label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('speedometer_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_model_on_sample(model, data_dir, transform):
    """Test the model on a few sample images"""
    model.eval()
    class_names = ['Stopped', 'Moving']
    
    # Test on stopped images
    stopped_dir = data_dir['stopped']
    if os.path.exists(stopped_dir):
        stopped_images = [f for f in os.listdir(stopped_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if stopped_images:
            sample_img = os.path.join(stopped_dir, stopped_images[0])
            image = Image.open(sample_img).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            print(f"Sample stopped image: {class_names[predicted_class]} (Confidence: {confidence:.4f})")
    
    # Test on moving images
    moving_dir = data_dir['moving']
    if os.path.exists(moving_dir):
        moving_images = [f for f in os.listdir(moving_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if moving_images:
            sample_img = os.path.join(moving_dir, moving_images[0])
            image = Image.open(sample_img).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            print(f"Sample moving image: {class_names[predicted_class]} (Confidence: {confidence:.4f})")

def main():
    # Define your data directories
    # Make sure you have this folder structure:
    # speedometer_data/
    # ├── stopped/         (images where needle is horizontal left - train stopped)
    # │   ├── stopped_1.jpg
    # │   ├── stopped_2.jpg
    # │   └── ...
    # └── moving/          (images where needle is not horizontal left - train moving)
    #     ├── moving_1.jpg
    #     ├── moving_2.jpg
    #     └── ...
    
    data_directories = {
        'stopped': 'train_stop_speedometer',        # Path to stopped train images
        'moving': 'train_moving_speedometer'           # Path to moving train images
    }
    
    # Check if directories exist
    for key, path in data_directories.items():
        if not os.path.exists(path):
            print(f"Error: Directory '{path}' does not exist!")
            print(f"Please create the directory structure and add your speedometer images.")
            return
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create dataset
    full_dataset = SpeedometerDataset(data_directories, transform=train_transform)
    
    if len(full_dataset) == 0:
        print("Error: No images found! Please check your data directories.")
        return
    
    # Split dataset (80% train, 20% validation)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)  # For reproducible splits
    )
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=16, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = EfficientNetSpeedometerDetector(
        model_name='efficientnet_b0', 
        num_classes=2
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\nStarting training...")
    train_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, 
        num_epochs=25, learning_rate=0.001
    )
    
    # Plot training history
    plot_training_history(train_losses, train_accuracies, val_accuracies)
    
    # Test model on sample images
    print("\nTesting model on sample images:")
    test_model_on_sample(model, data_directories, val_transform)
    
    print("\n" + "="*70)
    print("Training completed!")
    print(f"Best validation accuracy: {max(val_accuracies):.2f}%")
    print("Model saved as: best_speedometer_detection_model.pth")
    print("Training history plot saved as: speedometer_training_history.png")
    print("="*70)

if __name__ == "__main__":
    main()