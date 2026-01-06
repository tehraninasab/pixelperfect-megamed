# Per-Disease Performance Analysis for efficientnet-b0 (Pretrained)

## Overview

This report analyzes the performance of the CheXpert classifier using the efficientnet-b0 architecture (pretrained) across different image sizes for each disease in the dataset.

Analyzed image sizes: 1024

## Performance Across Image Sizes

### Cardiomegaly

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.4183 | 0.4259 | 0.4682 | 0.0270 | 0.0511 |

**Best image size for Cardiomegaly by F1 score**: 1024 (F1: 0.0511)

### Lung Opacity

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.5615 | 0.6025 | 0.7779 | 0.5464 | 0.6419 |

**Best image size for Lung Opacity by F1 score**: 1024 (F1: 0.6419)

### Edema

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.5346 | 0.5581 | 0.4988 | 0.6675 | 0.5710 |

**Best image size for Edema by F1 score**: 1024 (F1: 0.5710)

### No Finding

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8677 | 0.6507 | 0.2773 | 0.0088 | 0.0171 |

**Best image size for No Finding by F1 score**: 1024 (F1: 0.0171)

### Pneumothorax

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8765 | 0.6065 | 0.2558 | 0.0031 | 0.0062 |

**Best image size for Pneumothorax by F1 score**: 1024 (F1: 0.0062)

### Pleural Effusion

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.5131 | 0.6287 | 0.7143 | 0.0007 | 0.0014 |

**Best image size for Pleural Effusion by F1 score**: 1024 (F1: 0.0014)

## Best Image Size per Disease

| Disease | Best Image Size | F1 Score | AUROC |
| ------- | --------------- | -------- | ----- |
| Cardiomegaly | 1024 | 0.0511 | 0.4259 |
| Lung Opacity | 1024 | 0.6419 | 0.6025 |
| Edema | 1024 | 0.5710 | 0.5581 |
| No Finding | 1024 | 0.0171 | 0.6507 |
| Pneumothorax | 1024 | 0.0062 | 0.6065 |
| Pleural Effusion | 1024 | 0.0014 | 0.6287 |

## Conclusion

Based on the average F1 score across all diseases, the overall best image size for efficientnet-b0 (pretrained) is **1024**.

However, as shown in the per-disease analysis, the optimal image size varies across different pathologies. Consider using the disease-specific optimal image sizes if focusing on particular conditions.
