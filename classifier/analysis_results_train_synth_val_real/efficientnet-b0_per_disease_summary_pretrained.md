# Per-Disease Performance Analysis for efficientnet-b0 (Pretrained)

## Overview

This report analyzes the performance of the CheXpert classifier using the efficientnet-b0 architecture (pretrained) across different image sizes for each disease in the dataset.

Analyzed image sizes: 1024

## Performance Across Image Sizes

### Cardiomegaly

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.4168 | 0.4313 | 0.4697 | 0.0481 | 0.0872 |

**Best image size for Cardiomegaly by F1 score**: 1024 (F1: 0.0872)

### Lung Opacity

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.5298 | 0.5716 | 0.7631 | 0.5022 | 0.6057 |

**Best image size for Lung Opacity by F1 score**: 1024 (F1: 0.6057)

### Edema

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.5570 | 0.5904 | 0.5179 | 0.6515 | 0.5771 |

**Best image size for Edema by F1 score**: 1024 (F1: 0.5771)

### No Finding

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8687 | 0.6493 | 0.2833 | 0.0045 | 0.0090 |

**Best image size for No Finding by F1 score**: 1024 (F1: 0.0090)

### Pneumothorax

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8767 | 0.5567 | 0.1500 | 0.0009 | 0.0017 |

**Best image size for Pneumothorax by F1 score**: 1024 (F1: 0.0017)

### Pleural Effusion

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.5130 | 0.6267 | 0.6667 | 0.0001 | 0.0003 |

**Best image size for Pleural Effusion by F1 score**: 1024 (F1: 0.0003)

## Best Image Size per Disease

| Disease | Best Image Size | F1 Score | AUROC |
| ------- | --------------- | -------- | ----- |
| Cardiomegaly | 1024 | 0.0872 | 0.4313 |
| Lung Opacity | 1024 | 0.6057 | 0.5716 |
| Edema | 1024 | 0.5771 | 0.5904 |
| No Finding | 1024 | 0.0090 | 0.6493 |
| Pneumothorax | 1024 | 0.0017 | 0.5567 |
| Pleural Effusion | 1024 | 0.0003 | 0.6267 |

## Conclusion

Based on the average F1 score across all diseases, the overall best image size for efficientnet-b0 (pretrained) is **1024**.

However, as shown in the per-disease analysis, the optimal image size varies across different pathologies. Consider using the disease-specific optimal image sizes if focusing on particular conditions.
