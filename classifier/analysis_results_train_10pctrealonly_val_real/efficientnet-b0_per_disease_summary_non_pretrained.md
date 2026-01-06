# Per-Disease Performance Analysis for efficientnet-b0 (Pretrained)

## Overview

This report analyzes the performance of the CheXpert classifier using the efficientnet-b0 architecture (pretrained) across different image sizes for each disease in the dataset.

Analyzed image sizes: 1024

## Performance Across Image Sizes

### Cardiomegaly

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.7629 | 0.8376 | 0.8076 | 0.7758 | 0.7914 |

**Best image size for Cardiomegaly by F1 score**: 1024 (F1: 0.7914)

### Lung Opacity

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8293 | 0.8905 | 0.8937 | 0.8656 | 0.8795 |

**Best image size for Lung Opacity by F1 score**: 1024 (F1: 0.8795)

### Edema

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.7523 | 0.8331 | 0.7543 | 0.6912 | 0.7214 |

**Best image size for Edema by F1 score**: 1024 (F1: 0.7214)

### No Finding

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8863 | 0.9003 | 0.5666 | 0.5440 | 0.5551 |

**Best image size for No Finding by F1 score**: 1024 (F1: 0.5551)

### Pneumothorax

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8755 | 0.7458 | 0.4777 | 0.1429 | 0.2200 |

**Best image size for Pneumothorax by F1 score**: 1024 (F1: 0.2200)

### Pleural Effusion

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.7715 | 0.8467 | 0.7714 | 0.7544 | 0.7628 |

**Best image size for Pleural Effusion by F1 score**: 1024 (F1: 0.7628)

## Best Image Size per Disease

| Disease | Best Image Size | F1 Score | AUROC |
| ------- | --------------- | -------- | ----- |
| Cardiomegaly | 1024 | 0.7914 | 0.8376 |
| Lung Opacity | 1024 | 0.8795 | 0.8905 |
| Edema | 1024 | 0.7214 | 0.8331 |
| No Finding | 1024 | 0.5551 | 0.9003 |
| Pneumothorax | 1024 | 0.2200 | 0.7458 |
| Pleural Effusion | 1024 | 0.7628 | 0.8467 |

## Conclusion

Based on the average F1 score across all diseases, the overall best image size for efficientnet-b0 (pretrained) is **1024**.

However, as shown in the per-disease analysis, the optimal image size varies across different pathologies. Consider using the disease-specific optimal image sizes if focusing on particular conditions.
