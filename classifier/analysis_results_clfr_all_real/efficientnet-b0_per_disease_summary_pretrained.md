# Per-Disease Performance Analysis for efficientnet-b0 (Pretrained)

## Overview

This report analyzes the performance of the CheXpert classifier using the efficientnet-b0 architecture (pretrained) across different image sizes for each disease in the dataset.

Analyzed image sizes: 1024

## Performance Across Image Sizes

### Cardiomegaly

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8173 | 0.8917 | 0.8318 | 0.8584 | 0.8449 |

**Best image size for Cardiomegaly by F1 score**: 1024 (F1: 0.8449)

### Lung Opacity

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8763 | 0.9330 | 0.9027 | 0.9280 | 0.9152 |

**Best image size for Lung Opacity by F1 score**: 1024 (F1: 0.9152)

### Edema

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8093 | 0.8946 | 0.7985 | 0.7876 | 0.7930 |

**Best image size for Edema by F1 score**: 1024 (F1: 0.7930)

### No Finding

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.9306 | 0.9531 | 0.7624 | 0.6800 | 0.7188 |

**Best image size for No Finding by F1 score**: 1024 (F1: 0.7188)

### Pneumothorax

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.9176 | 0.9167 | 0.7186 | 0.5413 | 0.6175 |

**Best image size for Pneumothorax by F1 score**: 1024 (F1: 0.6175)

### Pleural Effusion

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 1024 | 0.8185 | 0.8964 | 0.7907 | 0.8531 | 0.8207 |

**Best image size for Pleural Effusion by F1 score**: 1024 (F1: 0.8207)

## Best Image Size per Disease

| Disease | Best Image Size | F1 Score | AUROC |
| ------- | --------------- | -------- | ----- |
| Cardiomegaly | 1024 | 0.8449 | 0.8917 |
| Lung Opacity | 1024 | 0.9152 | 0.9330 |
| Edema | 1024 | 0.7930 | 0.8946 |
| No Finding | 1024 | 0.7188 | 0.9531 |
| Pneumothorax | 1024 | 0.6175 | 0.9167 |
| Pleural Effusion | 1024 | 0.8207 | 0.8964 |

## Conclusion

Based on the average F1 score across all diseases, the overall best image size for efficientnet-b0 (pretrained) is **1024**.

However, as shown in the per-disease analysis, the optimal image size varies across different pathologies. Consider using the disease-specific optimal image sizes if focusing on particular conditions.
