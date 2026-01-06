# Per-Disease Performance Analysis for resnet50 (Pretrained)

## Overview

This report analyzes the performance of the CheXpert classifier using the resnet50 architecture (pretrained) across different image sizes for each disease in the dataset.

Analyzed image sizes: 64, 128, 256, 512, 1024

## Performance Across Image Sizes

### Cardiomegaly

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 64 | 0.7611 | 0.8323 | 0.7775 | 0.8234 | 0.7998 |
| 128 | 0.7724 | 0.8516 | 0.8279 | 0.7667 | 0.7961 |
| 256 | 0.7901 | 0.8643 | 0.8085 | 0.8357 | 0.8219 |
| 512 | 0.7929 | 0.8709 | 0.7872 | 0.8808 | 0.8314 |
| 1024 | 0.8042 | 0.8833 | 0.8369 | 0.8226 | 0.8297 |

**Best image size for Cardiomegaly by F1 score**: 512 (F1: 0.8314)

### Lung Opacity

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 64 | 0.8197 | 0.8731 | 0.8568 | 0.8998 | 0.8778 |
| 128 | 0.8313 | 0.8975 | 0.9015 | 0.8594 | 0.8799 |
| 256 | 0.8514 | 0.9099 | 0.8870 | 0.9092 | 0.8980 |
| 512 | 0.8520 | 0.9164 | 0.8617 | 0.9460 | 0.9019 |
| 1024 | 0.8688 | 0.9291 | 0.9040 | 0.9147 | 0.9093 |

**Best image size for Lung Opacity by F1 score**: 1024 (F1: 0.9093)

### Edema

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 64 | 0.7416 | 0.8196 | 0.7132 | 0.7409 | 0.7268 |
| 128 | 0.7494 | 0.8404 | 0.7930 | 0.6221 | 0.6972 |
| 256 | 0.7736 | 0.8563 | 0.7423 | 0.7840 | 0.7626 |
| 512 | 0.7786 | 0.8688 | 0.7319 | 0.8250 | 0.7757 |
| 1024 | 0.7967 | 0.8850 | 0.8081 | 0.7367 | 0.7707 |

**Best image size for Edema by F1 score**: 512 (F1: 0.7757)

### No Finding

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 64 | 0.8877 | 0.8835 | 0.6101 | 0.3840 | 0.4713 |
| 128 | 0.8743 | 0.9063 | 0.5137 | 0.6767 | 0.5841 |
| 256 | 0.9035 | 0.9221 | 0.6600 | 0.5371 | 0.5922 |
| 512 | 0.9068 | 0.9315 | 0.7999 | 0.3808 | 0.5160 |
| 1024 | 0.9222 | 0.9465 | 0.7303 | 0.6398 | 0.6821 |

**Best image size for No Finding by F1 score**: 1024 (F1: 0.6821)

### Pneumothorax

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 64 | 0.8665 | 0.7167 | 0.4039 | 0.1833 | 0.2522 |
| 128 | 0.8708 | 0.7442 | 0.4372 | 0.1802 | 0.2552 |
| 256 | 0.8807 | 0.8001 | 0.5248 | 0.3072 | 0.3875 |
| 512 | 0.8905 | 0.8121 | 0.6160 | 0.2882 | 0.3926 |
| 1024 | 0.9013 | 0.8700 | 0.6560 | 0.4123 | 0.5064 |

**Best image size for Pneumothorax by F1 score**: 1024 (F1: 0.5064)

### Pleural Effusion

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 64 | 0.7318 | 0.8053 | 0.7112 | 0.7567 | 0.7332 |
| 128 | 0.7581 | 0.8370 | 0.7255 | 0.8098 | 0.7653 |
| 256 | 0.7851 | 0.8650 | 0.7882 | 0.7641 | 0.7760 |
| 512 | 0.7970 | 0.8782 | 0.7622 | 0.8476 | 0.8026 |
| 1024 | 0.8077 | 0.8876 | 0.7898 | 0.8246 | 0.8068 |

**Best image size for Pleural Effusion by F1 score**: 1024 (F1: 0.8068)

## Best Image Size per Disease

| Disease | Best Image Size | F1 Score | AUROC |
| ------- | --------------- | -------- | ----- |
| Cardiomegaly | 512 | 0.8314 | 0.8709 |
| Lung Opacity | 1024 | 0.9093 | 0.9291 |
| Edema | 512 | 0.7757 | 0.8688 |
| No Finding | 1024 | 0.6821 | 0.9465 |
| Pneumothorax | 1024 | 0.5064 | 0.8700 |
| Pleural Effusion | 1024 | 0.8068 | 0.8876 |

## Conclusion

Based on the average F1 score across all diseases, the overall best image size for resnet50 (pretrained) is **1024**.

However, as shown in the per-disease analysis, the optimal image size varies across different pathologies. Consider using the disease-specific optimal image sizes if focusing on particular conditions.
