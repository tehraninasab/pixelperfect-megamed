# Per-Disease Performance Analysis for efficientnet-b0 (Pretrained)

## Overview

This report analyzes the performance of the CheXpert classifier using the efficientnet-b0 architecture (pretrained) across different image sizes for each disease in the dataset.

Analyzed image sizes: 1024

## Performance Across Image Sizes

### Cardiomegaly

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.4205 | 0.5514 | 1.0000 | 0.0002 | 0.0004 |

**Best image size for Cardiomegaly by F1 score**: 1024 (F1: 0.0004)

### Lung Opacity

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.2808 | 0.4755 | 0.5000 | 0.0001 | 0.0003 |

**Best image size for Lung Opacity by F1 score**: 1024 (F1: 0.0003)

### Edema

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.4640 | 0.5130 | 0.4639 | 0.9968 | 0.6331 |

**Best image size for Edema by F1 score**: 1024 (F1: 0.6331)

### No Finding

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8694 | 0.6644 | 0.2500 | 0.0005 | 0.0011 |

**Best image size for No Finding by F1 score**: 1024 (F1: 0.0011)

### Pneumothorax

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8768 | 0.6123 | 0.2000 | 0.0011 | 0.0023 |

**Best image size for Pneumothorax by F1 score**: 1024 (F1: 0.0023)

### Pleural Effusion

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.5129 | 0.5876 | 0.4667 | 0.0005 | 0.0010 |

**Best image size for Pleural Effusion by F1 score**: 1024 (F1: 0.0010)

## Best Image Size per Disease

| Disease | Best Image Size | F1 Score | AUROC |
| ------- | --------------- | -------- | ----- |
| Cardiomegaly | 1024 | 0.0004 | 0.5514 |
| Lung Opacity | 1024 | 0.0003 | 0.4755 |
| Edema | 1024 | 0.6331 | 0.5130 |
| No Finding | 1024 | 0.0011 | 0.6644 |
| Pneumothorax | 1024 | 0.0023 | 0.6123 |
| Pleural Effusion | 1024 | 0.0010 | 0.5876 |

## Conclusion

Based on the average F1 score across all diseases, the overall best image size for efficientnet-b0 (pretrained) is **1024**.

However, as shown in the per-disease analysis, the optimal image size varies across different pathologies. Consider using the disease-specific optimal image sizes if focusing on particular conditions.
