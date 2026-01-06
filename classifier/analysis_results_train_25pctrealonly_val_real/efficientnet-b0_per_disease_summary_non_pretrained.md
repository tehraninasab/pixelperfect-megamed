# Per-Disease Performance Analysis for efficientnet-b0 (Pretrained)

## Overview

This report analyzes the performance of the CheXpert classifier using the efficientnet-b0 architecture (pretrained) across different image sizes for each disease in the dataset.

Analyzed image sizes: 1024

## Performance Across Image Sizes

### Cardiomegaly

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.7967 | 0.8716 | 0.8166 | 0.8373 | 0.8268 |

**Best image size for Cardiomegaly by F1 score**: 1024 (F1: 0.8268)

### Lung Opacity

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8569 | 0.9171 | 0.8956 | 0.9068 | 0.9012 |

**Best image size for Lung Opacity by F1 score**: 1024 (F1: 0.9012)

### Edema

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.7848 | 0.8678 | 0.7691 | 0.7662 | 0.7676 |

**Best image size for Edema by F1 score**: 1024 (F1: 0.7676)

### No Finding

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.9105 | 0.9316 | 0.6782 | 0.5973 | 0.6352 |

**Best image size for No Finding by F1 score**: 1024 (F1: 0.6352)

### Pneumothorax

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8929 | 0.8434 | 0.6019 | 0.3768 | 0.4635 |

**Best image size for Pneumothorax by F1 score**: 1024 (F1: 0.4635)

### Pleural Effusion

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.7979 | 0.8756 | 0.7846 | 0.8064 | 0.7954 |

**Best image size for Pleural Effusion by F1 score**: 1024 (F1: 0.7954)

## Best Image Size per Disease

| Disease | Best Image Size | F1 Score | AUROC |
| ------- | --------------- | -------- | ----- |
| Cardiomegaly | 1024 | 0.8268 | 0.8716 |
| Lung Opacity | 1024 | 0.9012 | 0.9171 |
| Edema | 1024 | 0.7676 | 0.8678 |
| No Finding | 1024 | 0.6352 | 0.9316 |
| Pneumothorax | 1024 | 0.4635 | 0.8434 |
| Pleural Effusion | 1024 | 0.7954 | 0.8756 |

## Conclusion

Based on the average F1 score across all diseases, the overall best image size for efficientnet-b0 (pretrained) is **1024**.

However, as shown in the per-disease analysis, the optimal image size varies across different pathologies. Consider using the disease-specific optimal image sizes if focusing on particular conditions.
