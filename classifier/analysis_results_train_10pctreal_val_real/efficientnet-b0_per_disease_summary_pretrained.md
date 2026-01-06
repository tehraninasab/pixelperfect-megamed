# Per-Disease Performance Analysis for efficientnet-b0 (Pretrained)

## Overview

This report analyzes the performance of the CheXpert classifier using the efficientnet-b0 architecture (pretrained) across different image sizes for each disease in the dataset.

Analyzed image sizes: 1024

## Performance Across Image Sizes

### Cardiomegaly

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8006 | 0.8754 | 0.8163 | 0.8464 | 0.8311 |

**Best image size for Cardiomegaly by F1 score**: 1024 (F1: 0.8311)

### Lung Opacity

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8654 | 0.9229 | 0.9062 | 0.9066 | 0.9064 |

**Best image size for Lung Opacity by F1 score**: 1024 (F1: 0.9064)

### Edema

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.7913 | 0.8770 | 0.7851 | 0.7573 | 0.7710 |

**Best image size for Edema by F1 score**: 1024 (F1: 0.7710)

### No Finding

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.9210 | 0.9377 | 0.7372 | 0.6125 | 0.6691 |

**Best image size for No Finding by F1 score**: 1024 (F1: 0.6691)

### Pneumothorax

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.9020 | 0.8678 | 0.6730 | 0.3936 | 0.4967 |

**Best image size for Pneumothorax by F1 score**: 1024 (F1: 0.4967)

### Pleural Effusion

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.7964 | 0.8778 | 0.7953 | 0.7836 | 0.7894 |

**Best image size for Pleural Effusion by F1 score**: 1024 (F1: 0.7894)

## Best Image Size per Disease

| Disease | Best Image Size | F1 Score | AUROC |
| ------- | --------------- | -------- | ----- |
| Cardiomegaly | 1024 | 0.8311 | 0.8754 |
| Lung Opacity | 1024 | 0.9064 | 0.9229 |
| Edema | 1024 | 0.7710 | 0.8770 |
| No Finding | 1024 | 0.6691 | 0.9377 |
| Pneumothorax | 1024 | 0.4967 | 0.8678 |
| Pleural Effusion | 1024 | 0.7894 | 0.8778 |

## Conclusion

Based on the average F1 score across all diseases, the overall best image size for efficientnet-b0 (pretrained) is **1024**.

However, as shown in the per-disease analysis, the optimal image size varies across different pathologies. Consider using the disease-specific optimal image sizes if focusing on particular conditions.
