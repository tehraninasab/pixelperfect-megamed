# Per-Disease Performance Analysis for efficientnet-b0 (Pretrained)

## Overview

This report analyzes the performance of the CheXpert classifier using the efficientnet-b0 architecture (pretrained) across different image sizes for each disease in the dataset.

Analyzed image sizes: 1024

## Performance Across Image Sizes

### Cardiomegaly

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8198 | 0.8963 | 0.8246 | 0.8752 | 0.8492 |

**Best image size for Cardiomegaly by F1 score**: 1024 (F1: 0.8492)

### Lung Opacity

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8753 | 0.9355 | 0.8921 | 0.9403 | 0.9156 |

**Best image size for Lung Opacity by F1 score**: 1024 (F1: 0.9156)

### Edema

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8075 | 0.8964 | 0.7654 | 0.8437 | 0.8026 |

**Best image size for Edema by F1 score**: 1024 (F1: 0.8026)

### No Finding

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.9262 | 0.9518 | 0.7439 | 0.6623 | 0.7007 |

**Best image size for No Finding by F1 score**: 1024 (F1: 0.7007)

### Pneumothorax

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.9095 | 0.8990 | 0.7047 | 0.4530 | 0.5515 |

**Best image size for Pneumothorax by F1 score**: 1024 (F1: 0.5515)

### Pleural Effusion

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8156 | 0.8973 | 0.7813 | 0.8629 | 0.8200 |

**Best image size for Pleural Effusion by F1 score**: 1024 (F1: 0.8200)

## Best Image Size per Disease

| Disease | Best Image Size | F1 Score | AUROC |
| ------- | --------------- | -------- | ----- |
| Cardiomegaly | 1024 | 0.8492 | 0.8963 |
| Lung Opacity | 1024 | 0.9156 | 0.9355 |
| Edema | 1024 | 0.8026 | 0.8964 |
| No Finding | 1024 | 0.7007 | 0.9518 |
| Pneumothorax | 1024 | 0.5515 | 0.8990 |
| Pleural Effusion | 1024 | 0.8200 | 0.8973 |

## Conclusion

Based on the average F1 score across all diseases, the overall best image size for efficientnet-b0 (pretrained) is **1024**.

However, as shown in the per-disease analysis, the optimal image size varies across different pathologies. Consider using the disease-specific optimal image sizes if focusing on particular conditions.
