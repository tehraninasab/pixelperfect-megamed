import pandas as pd
import numpy as np
import os
import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_score, recall_score
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from classifier.train_chexpert import ChestXrayClassifier

import torch
from torchvision import models
import torch.nn as nn
import sys
sys.path.append('.')
import torch
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

TARGET_COLS = ["Cardiomegaly", "Lung Opacity", "Edema", "No Finding", "Pneumothorax", "Pleural Effusion"]
N_SAMPLES = 100000
root = '/usr/local/faststorage/datasets/mimic-cxr-jpg-2.0.0.physionet.org/'
# Make sure the ChestXrayClassifier and TARGET_COLS are already defined in your notebook

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv(os.path.join("datasets", "mimic-cxr", f'test_metadata_{N_SAMPLES}.csv'))
# metadata = pd.read_csv('/usr/local/faststorage/datasets/mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-metadata.csv')
# frontal_views = ["PA", "AP"]
# frontal_df = metadata[metadata["ViewPosition"].isin(frontal_views)]
# labels_df = pd.read_csv('/usr/local/faststorage/datasets/mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-chexpert.csv')
# df = frontal_df.merge(labels_df, on="subject_id", how="inner")
# df = df[df[TARGET_COLS].sum(axis=1) > 0]
# df = df.sample(n=N_SAMPLES, random_state=42)
# # Replace NaN and -1 with 0
# df[TARGET_COLS] = df[TARGET_COLS].replace([-1, np.nan], 0)
# df.to_csv(os.path.join("datasets", "mimic-cxr", f'test_metadata_{N_SAMPLES}.csv'), index=False)


def get_mimic_jpg_path(root, subject_id, study_id, dicom_id):
    subject_str = f"{int(subject_id):08d}"
    return os.path.join(
        root, "files",
        f"p{subject_str[:2]}",
        f"p{subject_str}",
        f"s{int(study_id)}",
        f"{dicom_id}.jpg"
    )

class ChestXrayDataset(Dataset):
    def __init__(self, metadata, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            root_dirs (dict): Dictionary with 'real' and 'synthetic' keys and image root dirs as values.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        self.annotations = metadata
        self.root_dir = root_dir
        self.transform = transform
        
        # Extract target columns (all except 'Path')
        self.target_cols = TARGET_COLS
        
        for col in self.target_cols:
            if col not in self.annotations.columns:
                print(f"Warning: Column '{col}' not found in annotations. Initializing with zeros.")
                self.annotations[col] = 0

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        
        subject_id = row['subject_id']
        study_id = row['study_id_x']
        dicom_id = row['dicom_id']
                
        img_path = get_mimic_jpg_path(self.root_dir, subject_id, study_id, dicom_id)
        
        # Using PIL for better compatibility
        image = Image.open(img_path).convert('RGB')


        if self.transform:
            image = self.transform(image)
        # Get all the target columns as a tensor
        labels = torch.tensor(row[self.target_cols].values.astype(np.float32))
        
        return image, labels
    
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

resolutions = [512, 256, 128, 64, 1024]
for res in resolutions:
    checkpoint_path = f'classifier/results_clfr_highres_rerun/efficientnet-b0_{res}_pretrained/chexpert_classifier.pth'
    # checkpoint_path = f'classifier/results_clfr_highres_augmented_v4_epoch_0/efficientnet-b0_{res}_pretrained/safe_model.pth'
    # output_dir = f"outputs/mimic-cxr-aug-{N_SAMPLES}"
    output_dir = f"outputs/{res}/mimic-cxr-noaug-{N_SAMPLES}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    
    
    transform = transforms.Compose([
            transforms.Resize((res, res)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
        ])

    test_dataset = ChestXrayDataset(
            metadata=df,
            root_dir=root,
            transform=transform
        )
    
    test_loader = DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=16
        )

    # Instantiate the model with the same architecture used during training
    model = ChestXrayClassifier(
        num_classes=len(TARGET_COLS),
        base_classifier='efficientnet-b0',  # or 'efficientnet-b0', 'densenet121'
        pretrained=True  # should match training config
    )

    state_dict = torch.load(checkpoint_path, map_location=device)
    # Load the checkpoint
    model.load_state_dict(state_dict)

    # Move model to the device and set to eval mode
    model = model.to(device)
    model.eval()

    print("Model loaded and ready for inference.")


    print("Evaluating the model on the test set...")
    # Evaluate the model
    _, _ = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        target_cols=TARGET_COLS,
        metrics_output_path=os.path.join(output_dir, 'test_metrics.csv'),
        debug=False
    )
    print(f"Evaluation completed successfully for resolution {res}!")
