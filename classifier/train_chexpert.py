import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_score, recall_score

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.io import read_image
from PIL import Image

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Constants
NUM_CLASSES = 6   # Six medical conditions to predict
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
TARGET_COLS = TARGET_COLS = ["Cardiomegaly", "Lung Opacity", "Edema", "No Finding", "Pneumothorax", "Pleural Effusion"]

class ChestXrayDataset(Dataset):
    def __init__(self, metadata, root_dirs, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            root_dirs (dict): Dictionary with 'real' and 'synthetic' keys and image root dirs as values.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations = metadata
        self.root_dirs = root_dirs
        self.transform = transform
        
        # Extract target columns (all except 'Path')
        self.target_cols = TARGET_COLS

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = os.path.join(self.root_dirs['real'], row["Path"].replace('CheXpert-v1.0', 'CheXpert-v1.0_1024x1024'))
        
        # Using PIL for better compatibility
        image = Image.open(img_path).convert('RGB')


        if self.transform:
            image = self.transform(image)
        # Get all the target columns as a tensor
        labels = torch.tensor(row[self.target_cols].values.astype(np.float32))
        
        return image, labels

class ChestXrayClassifier(nn.Module):
    def __init__(self, num_classes, base_classifier, pretrained):
        """
        Args:
            num_classes (int): Number of output classes.
            base_classifier (str): Base classifier architecture to use ('resnet50' or 'efficientnet-b0').
            pretrained (bool): Whether to use a pre-trained model.
        """
        super().__init__()

        
        # Load a pre-trained ResNet-50 model
        if base_classifier == 'resnet50':
            self.model = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            num_features = self.model.fc.in_features
            
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes),
                nn.Sigmoid()  # Sigmoid for multi-label classification
            )
            
        elif base_classifier == 'efficientnet-b0':
            self.model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            num_features = self.model.classifier[1].in_features
            
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(num_features, num_classes),
                nn.Sigmoid()  # Sigmoid for multi-label classification
            )
        elif base_classifier == 'densenet121':
            self.model = models.densenet121(
                weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
            num_features = self.model.classifier.in_features
            
            self.model.classifier = nn.Sequential(
                nn.Linear(num_features, num_classes),
                nn.Sigmoid()  # Sigmoid for multi-label classification
            )
    
    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, early_stopping_patience, debug=False):
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []
    steps = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for inputs, labels in train_progress:
            steps += 1
            if debug and steps > 10:
                break
            
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            train_progress.set_postfix(loss=loss.item())
        
        if debug and steps > 10:
            break
        # Calculate epoch loss
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        
        steps = 0
        with torch.no_grad():
            for inputs, labels in val_progress:
                steps += 1
                if debug and steps > 10:
                    break
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                val_progress.set_postfix(loss=loss.item())
        
        # Calculate epoch loss
        epoch_val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # Step the scheduler
        scheduler.step(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        
        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"✓ Validation loss decreased. Saving model...")
        else:
            patience_counter += 1
            print(f"! Validation loss did not decrease. Patience: {patience_counter}/{early_stopping_patience}")
            
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered!")
                break
    
    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

def evaluate_model(model, test_loader, device, target_cols, metrics_output_path, debug=False):

    model.eval()
    all_labels = []
    all_preds = []
    steps = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            steps += 1
            if debug and steps > 10:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    binarized_preds = (all_preds >= 0.5).astype(int)

    # --- Per-class metrics ---
    results = {
        "Class": [],
        "Accuracy": [],
        "AUROC": [],
        "Precision": [],
        "Recall": [],
        "F1": []
    }

    for i, class_name in enumerate(target_cols):
        y_true = all_labels[:, i]
        y_pred = binarized_preds[:, i]
        y_score = all_preds[:, i]

        results["Class"].append(class_name)
        results["Accuracy"].append(accuracy_score(y_true, y_pred))
        results["Precision"].append(precision_score(y_true, y_pred, zero_division=0))
        results["Recall"].append(recall_score(y_true, y_pred, zero_division=0))
        results["F1"].append(f1_score(y_true, y_pred, zero_division=0))

        if len(np.unique(y_true)) > 1:
            auc_score = roc_auc_score(y_true, y_score)
            results["AUROC"].append(auc_score)

            precision, recall, _ = precision_recall_curve(y_true, y_score)
            pr_auc = auc(recall, precision)
            print(f"{class_name} - ROC AUC: {auc_score:.4f}, PR AUC: {pr_auc:.4f}")
        else:
            results["AUROC"].append(np.nan)
            print(f"{class_name} - ROC AUC: N/A (Only one class present)")

    df = pd.DataFrame(results)

    # --- Macro-average row ---
    df.loc['Mean'] = df.mean(numeric_only=True)
    df.loc['Mean', 'Class'] = 'Mean'

    # --- Micro/global metrics ---
    flat_true = all_labels.flatten()
    flat_pred = binarized_preds.flatten()

    global_metrics = {
        "Class": "Global",
        "Accuracy": accuracy_score(flat_true, flat_pred),
        "AUROC": np.nan,  # AUROC is undefined when combining multiple classes this way
        "Precision": precision_score(flat_true, flat_pred, zero_division=0),
        "Recall": recall_score(flat_true, flat_pred, zero_division=0),
        "F1": f1_score(flat_true, flat_pred, zero_division=0),
    }
    df.loc['Global'] = global_metrics

    # Save metrics
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    df.to_csv(metrics_output_path, index=False)
    print(f"Metrics saved to {metrics_output_path}")

    return all_labels, all_preds


def main(args):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
    ])
    
    root_dirs = {
    "real": args.real_root,
    "synthetic": args.synthetic_root
    }

    train_df = pd.read_csv(args.train_csv)
    
    val_df = pd.read_csv(args.validation_csv)
    val_df['Source'] = 'real'
    
    # Get target columns
    print(f"Target columns: {TARGET_COLS}")
    
    
    train_dataset = ChestXrayDataset(
        metadata=train_df,
        root_dirs=root_dirs,
        transform=transform
    )
    
    val_dataset = ChestXrayDataset(
        metadata=val_df,
        root_dirs=root_dirs,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    test_df = pd.read_csv(args.test_csv)
    # Add a column Source and set it to 'real' for all rows
    test_df['Source'] = 'real'
    test_dataset = ChestXrayDataset(
        metadata=test_df,
        root_dirs=root_dirs,
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Testing samples: {len(test_df)}")
    print(f"Using image size: {args.image_size}x{args.image_size}")
    
    # Initialize model
    model = ChestXrayClassifier(num_classes=len(TARGET_COLS), base_classifier=args.base_classifier, pretrained=args.pretrained)
    print(f"Using base classifier: {args.base_classifier}")
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3,
    )
    
    # Train the model
    model, train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        debug=args.debug
    )
    
    # Plot training and validation losses in a single figure and save it in the output directory
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'losses.png'))
    plt.close()
    print("Training and validation losses plotted and saved.")

    # Save the model
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, 'chexpert_classifier.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    print("Training completed successfully!")
     
    print("Evaluating the model on the test set...")
    # Evaluate the model
    _, _ = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        target_cols=TARGET_COLS,
        metrics_output_path=os.path.join(args.output_dir, 'test_metrics.csv'),
        debug=args.debug
    )
    print("Evaluation completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a classifier for CheXpert dataset')
    parser.add_argument('--train_csv', type=str, default='/home/amarkr/dbpp/datasets/chexpert/chexpert_train.csv',
                        help='Path to the metadata CSV file')
    parser.add_argument('--validation_csv', type=str, default='datasets/chexpert/chexpert_val.csv',
                        help='Path to the validation CSV file')
    parser.add_argument('--test_csv', type=str, default='datasets/chexpert/chexpert_test.csv',
                help='Path to the test CSV file')
    parser.add_argument('--real_root', type=str, default=None,
                    help='Root directory of real images')
    parser.add_argument('--synthetic_root', type=str, default=None,
                    help='Root directory of synthetic images')
    parser.add_argument('--output_dir', type=str, default='saved_models/chexpert_classifier',
                        help='Directory to save the trained model')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS,
                        help='Number of epochs to train')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--base_classifier', type=str, default='resnet50',
                        choices=['resnet50', 'efficientnet-b0', 'densenet121'],
                        help='Base classifier architecture to use')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pre-trained model weights')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode for testing')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Size of the input images (will be resized to a square)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    # Save the arguments to a file
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    print(f"Arguments saved to {os.path.join(args.output_dir, 'args.txt')}")
    # Run the main function
    main(args)