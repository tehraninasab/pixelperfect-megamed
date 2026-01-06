
import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score 

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# Constants
IMAGE_SIZE = 224
NUM_CLASSES = 8  
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

class ISIC2019Dataset(Dataset):
    def __init__(self, metadata, root_dirs, transform=None):
        self.annotations = metadata
        self.root_dirs = root_dirs
        self.transform = transform
        self.labels = metadata[CLASS_NAMES].values.argmax(axis=1)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        root = self.root_dirs[row["Source"]]
        

        if row["Source"] == "real":
            try:
                image_path = os.path.join(root, row["image"] + ".jpg")
                image = Image.open(image_path).convert('RGB')
            except FileNotFoundError:
                # Try the downsampled version
                image_name = row['image'] + '_downsampled.jpg'
                image_path = os.path.join(root, image_name)
                image = Image.open(image_path).convert('RGB')
        else:
            image_path = os.path.join(root, row["image"])
            image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

class ISIC2019Classifier(nn.Module):
    def __init__(self, num_classes, base_classifier, pretrained):
        super(ISIC2019Classifier, self).__init__()
        if base_classifier == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        elif base_classifier == 'efficientnet-b0':
            self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            num_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(num_features, num_classes)
            )
        elif base_classifier == 'inception_v3':
            self.model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT if pretrained else None)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        else:
            raise ValueError("Unsupported base classifier. Choose from ['resnet50', 'efficientnet-b0', 'inception_v3']")

    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, patience, debug=False):
    best_val_loss, best_model_state, patience_counter = float('inf'), None, 0
    train_losses, val_losses = [], []
    steps = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels in train_progress:
            steps += 1
            if debug and steps > 10:
                break
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            train_progress.set_postfix(loss=loss.item())
        
        if debug and steps > 10:
            break
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
        epoch_val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model, train_losses, val_losses

def evaluate_model(model, test_loader, device, class_names, metrics_output_path, debug=False):
    model.eval()
    all_labels, all_preds = [], []
    all_probs = []
    steps = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            steps += 1
            if debug and steps > 10:
                break
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
     # Compute AUROC per class
    aurocs = []
    for i in range(len(class_names)):
        try:
            auroc = roc_auc_score((all_labels == i).astype(int), all_probs[:, i])
        except ValueError:
            auroc = float('nan')  # 👈 Handle cases with only one class present
        aurocs.append(auroc)

    # Accuracy (overall)
    overall_acc = accuracy_score(all_labels, all_preds)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(len(class_names)), zero_division=0
    )

    # Per-class accuracy
    class_correct = np.zeros(len(class_names))
    class_total = np.zeros(len(class_names))
    for true, pred in zip(all_labels, all_preds):
        class_total[true] += 1
        if true == pred:
            class_correct[true] += 1
    class_accuracy = class_correct / np.maximum(class_total, 1)

    # DataFrame of per-class metrics
    df = pd.DataFrame({
        "Class": class_names,
        "Accuracy": class_accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "AUROC": aurocs,
        "Support": support
    })
    
    # Macro/micro AUROC
    try:
        macro_auroc = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovr')  # 👈
        micro_auroc = roc_auc_score(all_labels, all_probs, average='micro', multi_class='ovr')  # 👈
        weighted_auroc = roc_auc_score(all_labels, all_probs, average='weighted', multi_class='ovr')
    except ValueError:
        macro_auroc = micro_auroc = float('nan')

    # Macro-average (unweighted mean across classes)
    macro_avg = {
        "Class": "Macro Average",
        "Accuracy": class_accuracy.mean(),
        "Precision": precision.mean(),
        "Recall": recall.mean(),
        "F1-Score": f1.mean(),
        "AUROC": macro_auroc,
        "Support": support.sum()
    }

    # Micro-average (global precision/recall/F1 via total TP/FP/FN)
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='micro', zero_division=0
    )
    micro_avg = {
        "Class": "Micro Average",
        "Accuracy": overall_acc,
        "Precision": micro_precision,
        "Recall": micro_recall,
        "F1-Score": micro_f1,
        "AUROC": micro_auroc,
        "Support": support.sum()
    }

    weighted_avg = {
        "Class": "Weighted Average",
        "Accuracy": "",
        "Precision": "",
        "Recall": "",
        "F1-Score": "",
        "AUROC": weighted_auroc,
        "Support": support.sum()
    }

    # Global row (only overall accuracy is meaningful here)
    global_avg = {
        "Class": "Global",
        "Accuracy": overall_acc,
        "Precision": "",
        "Recall": "",
        "F1-Score": "",
        "Support": support.sum()
    }

    df = pd.concat([df, pd.DataFrame([macro_avg, micro_avg, weighted_avg, global_avg])], ignore_index=True)

    # Save metrics
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    df.to_csv(metrics_output_path, index=False)
    print(f"Saved metrics to {metrics_output_path}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(os.path.dirname(metrics_output_path), "confusion_matrix.png"))
    plt.close()
    print(f"Saved confusion matrix to {os.path.join(os.path.dirname(metrics_output_path), 'confusion_matrix.png')}")

    return all_labels.tolist(), all_preds.tolist()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    root_dirs = {"real": args.real_root, "synthetic": args.synthetic_root}
    for split in ["train", "validation", "test"]:
        df = pd.read_csv(getattr(args, f"{split}_csv"))
        setattr(args, f"{split}_df", df)
        if split in ['validation', 'test']:
            df['Source'] = 'real'

    train_dataset = ISIC2019Dataset(args.train_df, root_dirs, transform)
    val_dataset = ISIC2019Dataset(args.validation_df, root_dirs, transform)
    test_dataset = ISIC2019Dataset(args.test_df, root_dirs, transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ISIC2019Classifier(NUM_CLASSES, args.base_classifier, args.pretrained).to(device)
    # class_weights = [train_dataset.labels.value_counts().max() / train_dataset.annotations['level'].value_counts()[i] for i in range(NUM_CLASSES)]
    # class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    model, train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args.num_epochs, EARLY_STOPPING_PATIENCE, args.debug)

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "isic2019_classifier.pth"))

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.savefig(os.path.join(args.output_dir, 'loss_curve.png'))
    plt.close()

    evaluate_model(model, test_loader, device, CLASS_NAMES, os.path.join(args.output_dir, 'test_metrics.csv'), args.debug)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--validation_csv', type=str, required=True)
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--real_root', type=str, required=True)
    parser.add_argument('--synthetic_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='saved_models/isic2019_classifier')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--base_classifier', type=str, default='resnet50', choices=['resnet50', 'efficientnet-b0'])
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args)
