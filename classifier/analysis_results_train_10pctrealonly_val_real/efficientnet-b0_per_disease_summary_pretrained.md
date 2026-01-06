# Per-Disease Performance Analysis for efficientnet-b0 (Pretrained)

## Overview

This report analyzes the performance of the CheXpert classifier using the efficientnet-b0 architecture (pretrained) across different image sizes for each disease in the dataset.

Analyzed image sizes: 1024

## Performance Across Image Sizes

### Cardiomegaly

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.7994 | 0.8772 | 0.8258 | 0.8287 | 0.8273 |

**Best image size for Cardiomegaly by F1 score**: 1024 (F1: 0.8273)

### Lung Opacity

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8645 | 0.9228 | 0.8997 | 0.9135 | 0.9065 |

**Best image size for Lung Opacity by F1 score**: 1024 (F1: 0.9065)

### Edema

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.7854 | 0.8760 | 0.7950 | 0.7240 | 0.7579 |

**Best image size for Edema by F1 score**: 1024 (F1: 0.7579)

### No Finding

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.9199 | 0.9392 | 0.7148 | 0.6420 | 0.6764 |

**Best image size for No Finding by F1 score**: 1024 (F1: 0.6764)

### Pneumothorax

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.9007 | 0.8636 | 0.6542 | 0.4069 | 0.5018 |

**Best image size for Pneumothorax by F1 score**: 1024 (F1: 0.5018)

### Pleural Effusion

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.7993 | 0.8786 | 0.7913 | 0.7986 | 0.7949 |

**Best image size for Pleural Effusion by F1 score**: 1024 (F1: 0.7949)

## Best Image Size per Disease

| Disease | Best Image Size | F1 Score | AUROC |
| ------- | --------------- | -------- | ----- |
| Cardiomegaly | 1024 | 0.8273 | 0.8772 |
| Lung Opacity | 1024 | 0.9065 | 0.9228 |
| Edema | 1024 | 0.7579 | 0.8760 |
| No Finding | 1024 | 0.6764 | 0.9392 |
| Pneumothorax | 1024 | 0.5018 | 0.8636 |
| Pleural Effusion | 1024 | 0.7949 | 0.8786 |

## Conclusion

Based on the average F1 score across all diseases, the overall best image size for efficientnet-b0 (pretrained) is **1024**.

However, as shown in the per-disease analysis, the optimal image size varies across different pathologies. Consider using the disease-specific optimal image sizes if focusing on particular conditions.
