# Per-Disease Performance Analysis for efficientnet-b0 (Pretrained)

## Overview

This report analyzes the performance of the CheXpert classifier using the efficientnet-b0 architecture (pretrained) across different image sizes for each disease in the dataset.

Analyzed image sizes: 1024

## Performance Across Image Sizes

### Cardiomegaly

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8072 | 0.8833 | 0.8235 | 0.8493 | 0.8362 |

**Best image size for Cardiomegaly by F1 score**: 1024 (F1: 0.8362)

### Lung Opacity

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8670 | 0.9266 | 0.9012 | 0.9155 | 0.9083 |

**Best image size for Lung Opacity by F1 score**: 1024 (F1: 0.9083)

### Edema

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.7980 | 0.8821 | 0.7860 | 0.7758 | 0.7809 |

**Best image size for Edema by F1 score**: 1024 (F1: 0.7809)

### No Finding

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.9223 | 0.9425 | 0.7379 | 0.6275 | 0.6782 |

**Best image size for No Finding by F1 score**: 1024 (F1: 0.6782)

### Pneumothorax

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.9085 | 0.8930 | 0.6907 | 0.4621 | 0.5537 |

**Best image size for Pneumothorax by F1 score**: 1024 (F1: 0.5537)

### Pleural Effusion

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8071 | 0.8861 | 0.7827 | 0.8360 | 0.8085 |

**Best image size for Pleural Effusion by F1 score**: 1024 (F1: 0.8085)

## Best Image Size per Disease

| Disease | Best Image Size | F1 Score | AUROC |
| ------- | --------------- | -------- | ----- |
| Cardiomegaly | 1024 | 0.8362 | 0.8833 |
| Lung Opacity | 1024 | 0.9083 | 0.9266 |
| Edema | 1024 | 0.7809 | 0.8821 |
| No Finding | 1024 | 0.6782 | 0.9425 |
| Pneumothorax | 1024 | 0.5537 | 0.8930 |
| Pleural Effusion | 1024 | 0.8085 | 0.8861 |

## Conclusion

Based on the average F1 score across all diseases, the overall best image size for efficientnet-b0 (pretrained) is **1024**.

However, as shown in the per-disease analysis, the optimal image size varies across different pathologies. Consider using the disease-specific optimal image sizes if focusing on particular conditions.
