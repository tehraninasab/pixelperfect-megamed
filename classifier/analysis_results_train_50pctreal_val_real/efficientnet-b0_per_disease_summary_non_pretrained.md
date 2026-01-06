# Per-Disease Performance Analysis for efficientnet-b0 (Pretrained)

## Overview

This report analyzes the performance of the CheXpert classifier using the efficientnet-b0 architecture (pretrained) across different image sizes for each disease in the dataset.

Analyzed image sizes: 1024

## Performance Across Image Sizes

### Cardiomegaly

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8155 | 0.8912 | 0.8228 | 0.8687 | 0.8451 |

**Best image size for Cardiomegaly by F1 score**: 1024 (F1: 0.8451)

### Lung Opacity

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8721 | 0.9313 | 0.8939 | 0.9329 | 0.9130 |

**Best image size for Lung Opacity by F1 score**: 1024 (F1: 0.9130)

### Edema

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8047 | 0.8901 | 0.7951 | 0.7799 | 0.7874 |

**Best image size for Edema by F1 score**: 1024 (F1: 0.7874)

### No Finding

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.9208 | 0.9452 | 0.7591 | 0.5759 | 0.6549 |

**Best image size for No Finding by F1 score**: 1024 (F1: 0.6549)

### Pneumothorax

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.9052 | 0.8862 | 0.7004 | 0.3987 | 0.5081 |

**Best image size for Pneumothorax by F1 score**: 1024 (F1: 0.5081)

### Pleural Effusion

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8120 | 0.8927 | 0.7916 | 0.8335 | 0.8120 |

**Best image size for Pleural Effusion by F1 score**: 1024 (F1: 0.8120)

## Best Image Size per Disease

| Disease | Best Image Size | F1 Score | AUROC |
| ------- | --------------- | -------- | ----- |
| Cardiomegaly | 1024 | 0.8451 | 0.8912 |
| Lung Opacity | 1024 | 0.9130 | 0.9313 |
| Edema | 1024 | 0.7874 | 0.8901 |
| No Finding | 1024 | 0.6549 | 0.9452 |
| Pneumothorax | 1024 | 0.5081 | 0.8862 |
| Pleural Effusion | 1024 | 0.8120 | 0.8927 |

## Conclusion

Based on the average F1 score across all diseases, the overall best image size for efficientnet-b0 (pretrained) is **1024**.

However, as shown in the per-disease analysis, the optimal image size varies across different pathologies. Consider using the disease-specific optimal image sizes if focusing on particular conditions.
