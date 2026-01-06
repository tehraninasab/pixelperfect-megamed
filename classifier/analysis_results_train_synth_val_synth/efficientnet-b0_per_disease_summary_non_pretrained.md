# Per-Disease Performance Analysis for efficientnet-b0 (Pretrained)

## Overview

This report analyzes the performance of the CheXpert classifier using the efficientnet-b0 architecture (pretrained) across different image sizes for each disease in the dataset.

Analyzed image sizes: 1024

## Performance Across Image Sizes

### Cardiomegaly

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.4204 | 0.5806 | 0.5000 | 0.0002 | 0.0004 |

**Best image size for Cardiomegaly by F1 score**: 1024 (F1: 0.0004)

### Lung Opacity

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.2809 | 0.4216 | 0.5349 | 0.0011 | 0.0022 |

**Best image size for Lung Opacity by F1 score**: 1024 (F1: 0.0022)

### Edema

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.4635 | 0.4569 | 0.4548 | 0.7872 | 0.5765 |

**Best image size for Edema by F1 score**: 1024 (F1: 0.5765)

### No Finding

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8413 | 0.5388 | 0.1113 | 0.0310 | 0.0485 |

**Best image size for No Finding by F1 score**: 1024 (F1: 0.0485)

### Pneumothorax

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.7260 | 0.6173 | 0.1783 | 0.3410 | 0.2341 |

**Best image size for Pneumothorax by F1 score**: 1024 (F1: 0.2341)

### Pleural Effusion

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.5129 | 0.5796 | 0.4545 | 0.0004 | 0.0007 |

**Best image size for Pleural Effusion by F1 score**: 1024 (F1: 0.0007)

## Best Image Size per Disease

| Disease | Best Image Size | F1 Score | AUROC |
| ------- | --------------- | -------- | ----- |
| Cardiomegaly | 1024 | 0.0004 | 0.5806 |
| Lung Opacity | 1024 | 0.0022 | 0.4216 |
| Edema | 1024 | 0.5765 | 0.4569 |
| No Finding | 1024 | 0.0485 | 0.5388 |
| Pneumothorax | 1024 | 0.2341 | 0.6173 |
| Pleural Effusion | 1024 | 0.0007 | 0.5796 |

## Conclusion

Based on the average F1 score across all diseases, the overall best image size for efficientnet-b0 (pretrained) is **1024**.

However, as shown in the per-disease analysis, the optimal image size varies across different pathologies. Consider using the disease-specific optimal image sizes if focusing on particular conditions.
