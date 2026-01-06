# Per-Disease Performance Analysis for efficientnet-b0 (Pretrained)

## Overview

This report analyzes the performance of the CheXpert classifier using the efficientnet-b0 architecture (pretrained) across different image sizes for each disease in the dataset.

Analyzed image sizes: 1024

## Performance Across Image Sizes

### Cardiomegaly

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.4208 | 0.5027 | 0.7083 | 0.0010 | 0.0020 |

**Best image size for Cardiomegaly by F1 score**: 1024 (F1: 0.0020)

### Lung Opacity

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.3107 | 0.6528 | 0.8275 | 0.0526 | 0.0989 |

**Best image size for Lung Opacity by F1 score**: 1024 (F1: 0.0989)

### Edema

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.4754 | 0.5596 | 0.4686 | 0.9755 | 0.6331 |

**Best image size for Edema by F1 score**: 1024 (F1: 0.6331)

### No Finding

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8684 | 0.6446 | 0.4569 | 0.0468 | 0.0850 |

**Best image size for No Finding by F1 score**: 1024 (F1: 0.0850)

### Pneumothorax

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8770 | 0.6324 | 0.1429 | 0.0003 | 0.0006 |

**Best image size for Pneumothorax by F1 score**: 1024 (F1: 0.0006)

### Pleural Effusion

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.5168 | 0.6684 | 0.6961 | 0.0141 | 0.0277 |

**Best image size for Pleural Effusion by F1 score**: 1024 (F1: 0.0277)

## Best Image Size per Disease

| Disease | Best Image Size | F1 Score | AUROC |
| ------- | --------------- | -------- | ----- |
| Cardiomegaly | 1024 | 0.0020 | 0.5027 |
| Lung Opacity | 1024 | 0.0989 | 0.6528 |
| Edema | 1024 | 0.6331 | 0.5596 |
| No Finding | 1024 | 0.0850 | 0.6446 |
| Pneumothorax | 1024 | 0.0006 | 0.6324 |
| Pleural Effusion | 1024 | 0.0277 | 0.6684 |

## Conclusion

Based on the average F1 score across all diseases, the overall best image size for efficientnet-b0 (pretrained) is **1024**.

However, as shown in the per-disease analysis, the optimal image size varies across different pathologies. Consider using the disease-specific optimal image sizes if focusing on particular conditions.
