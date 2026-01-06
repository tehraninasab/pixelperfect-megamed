# Per-Disease Performance Analysis for efficientnet-b0 (Pretrained)

## Overview

This report analyzes the performance of the CheXpert classifier using the efficientnet-b0 architecture (pretrained) across different image sizes for each disease in the dataset.

Analyzed image sizes: 1024

## Performance Across Image Sizes

### Cardiomegaly

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8054 | 0.8800 | 0.8198 | 0.8513 | 0.8352 |

**Best image size for Cardiomegaly by F1 score**: 1024 (F1: 0.8352)

### Lung Opacity

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8670 | 0.9237 | 0.8893 | 0.9310 | 0.9097 |

**Best image size for Lung Opacity by F1 score**: 1024 (F1: 0.9097)

### Edema

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.7922 | 0.8790 | 0.7833 | 0.7632 | 0.7732 |

**Best image size for Edema by F1 score**: 1024 (F1: 0.7732)

### No Finding

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.9251 | 0.9437 | 0.7738 | 0.6016 | 0.6769 |

**Best image size for No Finding by F1 score**: 1024 (F1: 0.6769)

### Pneumothorax

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.9066 | 0.8886 | 0.6554 | 0.5058 | 0.5710 |

**Best image size for Pneumothorax by F1 score**: 1024 (F1: 0.5710)

### Pleural Effusion

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8022 | 0.8815 | 0.7862 | 0.8156 | 0.8007 |

**Best image size for Pleural Effusion by F1 score**: 1024 (F1: 0.8007)

## Best Image Size per Disease

| Disease | Best Image Size | F1 Score | AUROC |
| ------- | --------------- | -------- | ----- |
| Cardiomegaly | 1024 | 0.8352 | 0.8800 |
| Lung Opacity | 1024 | 0.9097 | 0.9237 |
| Edema | 1024 | 0.7732 | 0.8790 |
| No Finding | 1024 | 0.6769 | 0.9437 |
| Pneumothorax | 1024 | 0.5710 | 0.8886 |
| Pleural Effusion | 1024 | 0.8007 | 0.8815 |

## Conclusion

Based on the average F1 score across all diseases, the overall best image size for efficientnet-b0 (pretrained) is **1024**.

However, as shown in the per-disease analysis, the optimal image size varies across different pathologies. Consider using the disease-specific optimal image sizes if focusing on particular conditions.
