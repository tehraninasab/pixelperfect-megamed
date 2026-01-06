# Per-Disease Performance Analysis for efficientnet-b0 (Pretrained)

## Overview

This report analyzes the performance of the CheXpert classifier using the efficientnet-b0 architecture (pretrained) across different image sizes for each disease in the dataset.

Analyzed image sizes: 1024

## Performance Across Image Sizes

### Cardiomegaly

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.7883 | 0.8654 | 0.7790 | 0.8861 | 0.8291 |

**Best image size for Cardiomegaly by F1 score**: 1024 (F1: 0.8291)

### Lung Opacity

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8458 | 0.9101 | 0.8565 | 0.9438 | 0.8980 |

**Best image size for Lung Opacity by F1 score**: 1024 (F1: 0.8980)

### Edema

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.7819 | 0.8677 | 0.7383 | 0.8210 | 0.7774 |

**Best image size for Edema by F1 score**: 1024 (F1: 0.7774)

### No Finding

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8973 | 0.9151 | 0.7240 | 0.3433 | 0.4658 |

**Best image size for No Finding by F1 score**: 1024 (F1: 0.4658)

### Pneumothorax

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8816 | 0.7847 | 0.7124 | 0.0605 | 0.1116 |

**Best image size for Pneumothorax by F1 score**: 1024 (F1: 0.1116)

### Pleural Effusion

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.7626 | 0.8508 | 0.7105 | 0.8649 | 0.7801 |

**Best image size for Pleural Effusion by F1 score**: 1024 (F1: 0.7801)

## Best Image Size per Disease

| Disease | Best Image Size | F1 Score | AUROC |
| ------- | --------------- | -------- | ----- |
| Cardiomegaly | 1024 | 0.8291 | 0.8654 |
| Lung Opacity | 1024 | 0.8980 | 0.9101 |
| Edema | 1024 | 0.7774 | 0.8677 |
| No Finding | 1024 | 0.4658 | 0.9151 |
| Pneumothorax | 1024 | 0.1116 | 0.7847 |
| Pleural Effusion | 1024 | 0.7801 | 0.8508 |

## Conclusion

Based on the average F1 score across all diseases, the overall best image size for efficientnet-b0 (pretrained) is **1024**.

However, as shown in the per-disease analysis, the optimal image size varies across different pathologies. Consider using the disease-specific optimal image sizes if focusing on particular conditions.
