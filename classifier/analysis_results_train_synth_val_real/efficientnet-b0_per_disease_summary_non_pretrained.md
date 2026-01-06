# Per-Disease Performance Analysis for efficientnet-b0 (Pretrained)

## Overview

This report analyzes the performance of the CheXpert classifier using the efficientnet-b0 architecture (pretrained) across different image sizes for each disease in the dataset.

Analyzed image sizes: 1024

## Performance Across Image Sizes

### Cardiomegaly

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.4201 | 0.5148 | 0.4913 | 0.0169 | 0.0327 |

**Best image size for Cardiomegaly by F1 score**: 1024 (F1: 0.0327)

### Lung Opacity

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.2800 | 0.4180 | 0.3425 | 0.0012 | 0.0024 |

**Best image size for Lung Opacity by F1 score**: 1024 (F1: 0.0024)

### Edema

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.4575 | 0.4669 | 0.4501 | 0.7639 | 0.5665 |

**Best image size for Edema by F1 score**: 1024 (F1: 0.5665)

### No Finding

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8689 | 0.6670 | 0.3276 | 0.0051 | 0.0100 |

**Best image size for No Finding by F1 score**: 1024 (F1: 0.0100)

### Pneumothorax

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.6799 | 0.6174 | 0.1712 | 0.4180 | 0.2429 |

**Best image size for Pneumothorax by F1 score**: 1024 (F1: 0.2429)

### Pleural Effusion

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.5129 | 0.4737 | 0.3333 | 0.0001 | 0.0001 |

**Best image size for Pleural Effusion by F1 score**: 1024 (F1: 0.0001)

## Best Image Size per Disease

| Disease | Best Image Size | F1 Score | AUROC |
| ------- | --------------- | -------- | ----- |
| Cardiomegaly | 1024 | 0.0327 | 0.5148 |
| Lung Opacity | 1024 | 0.0024 | 0.4180 |
| Edema | 1024 | 0.5665 | 0.4669 |
| No Finding | 1024 | 0.0100 | 0.6670 |
| Pneumothorax | 1024 | 0.2429 | 0.6174 |
| Pleural Effusion | 1024 | 0.0001 | 0.4737 |

## Conclusion

Based on the average F1 score across all diseases, the overall best image size for efficientnet-b0 (pretrained) is **1024**.

However, as shown in the per-disease analysis, the optimal image size varies across different pathologies. Consider using the disease-specific optimal image sizes if focusing on particular conditions.
