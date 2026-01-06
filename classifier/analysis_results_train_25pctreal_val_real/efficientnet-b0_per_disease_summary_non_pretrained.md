# Per-Disease Performance Analysis for efficientnet-b0 (Pretrained)

## Overview

This report analyzes the performance of the CheXpert classifier using the efficientnet-b0 architecture (pretrained) across different image sizes for each disease in the dataset.

Analyzed image sizes: 1024

## Performance Across Image Sizes

### Cardiomegaly

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8027 | 0.8782 | 0.8234 | 0.8396 | 0.8314 |

**Best image size for Cardiomegaly by F1 score**: 1024 (F1: 0.8314)

### Lung Opacity

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8618 | 0.9225 | 0.9012 | 0.9073 | 0.9043 |

**Best image size for Lung Opacity by F1 score**: 1024 (F1: 0.9043)

### Edema

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.7903 | 0.8773 | 0.7446 | 0.8339 | 0.7867 |

**Best image size for Edema by F1 score**: 1024 (F1: 0.7867)

### No Finding

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.9101 | 0.9282 | 0.7238 | 0.5020 | 0.5928 |

**Best image size for No Finding by F1 score**: 1024 (F1: 0.5928)

### Pneumothorax

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8956 | 0.8468 | 0.6281 | 0.3686 | 0.4645 |

**Best image size for Pneumothorax by F1 score**: 1024 (F1: 0.4645)

### Pleural Effusion

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.7962 | 0.8778 | 0.8019 | 0.7724 | 0.7869 |

**Best image size for Pleural Effusion by F1 score**: 1024 (F1: 0.7869)

## Best Image Size per Disease

| Disease | Best Image Size | F1 Score | AUROC |
| ------- | --------------- | -------- | ----- |
| Cardiomegaly | 1024 | 0.8314 | 0.8782 |
| Lung Opacity | 1024 | 0.9043 | 0.9225 |
| Edema | 1024 | 0.7867 | 0.8773 |
| No Finding | 1024 | 0.5928 | 0.9282 |
| Pneumothorax | 1024 | 0.4645 | 0.8468 |
| Pleural Effusion | 1024 | 0.7869 | 0.8778 |

## Conclusion

Based on the average F1 score across all diseases, the overall best image size for efficientnet-b0 (pretrained) is **1024**.

However, as shown in the per-disease analysis, the optimal image size varies across different pathologies. Consider using the disease-specific optimal image sizes if focusing on particular conditions.
