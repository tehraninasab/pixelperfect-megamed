# Per-Disease Performance Analysis for efficientnet-b0 (Pretrained)

## Overview

This report analyzes the performance of the CheXpert classifier using the efficientnet-b0 architecture (pretrained) across different image sizes for each disease in the dataset.

Analyzed image sizes: 1024

## Performance Across Image Sizes

### Cardiomegaly

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8110 | 0.8864 | 0.8251 | 0.8551 | 0.8398 |

**Best image size for Cardiomegaly by F1 score**: 1024 (F1: 0.8398)

### Lung Opacity

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8697 | 0.9279 | 0.9002 | 0.9209 | 0.9104 |

**Best image size for Lung Opacity by F1 score**: 1024 (F1: 0.9104)

### Edema

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8019 | 0.8872 | 0.7981 | 0.7672 | 0.7823 |

**Best image size for Edema by F1 score**: 1024 (F1: 0.7823)

### No Finding

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.9264 | 0.9459 | 0.7554 | 0.6446 | 0.6956 |

**Best image size for No Finding by F1 score**: 1024 (F1: 0.6956)

### Pneumothorax

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.9103 | 0.9019 | 0.6736 | 0.5232 | 0.5889 |

**Best image size for Pneumothorax by F1 score**: 1024 (F1: 0.5889)

### Pleural Effusion

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8110 | 0.8896 | 0.7930 | 0.8282 | 0.8102 |

**Best image size for Pleural Effusion by F1 score**: 1024 (F1: 0.8102)

## Best Image Size per Disease

| Disease | Best Image Size | F1 Score | AUROC |
| ------- | --------------- | -------- | ----- |
| Cardiomegaly | 1024 | 0.8398 | 0.8864 |
| Lung Opacity | 1024 | 0.9104 | 0.9279 |
| Edema | 1024 | 0.7823 | 0.8872 |
| No Finding | 1024 | 0.6956 | 0.9459 |
| Pneumothorax | 1024 | 0.5889 | 0.9019 |
| Pleural Effusion | 1024 | 0.8102 | 0.8896 |

## Conclusion

Based on the average F1 score across all diseases, the overall best image size for efficientnet-b0 (pretrained) is **1024**.

However, as shown in the per-disease analysis, the optimal image size varies across different pathologies. Consider using the disease-specific optimal image sizes if focusing on particular conditions.
