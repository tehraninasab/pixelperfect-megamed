# Per-Disease Performance Analysis for efficientnet-b0 (Pretrained)

## Overview

This report analyzes the performance of the CheXpert classifier using the efficientnet-b0 architecture (pretrained) across different image sizes for each disease in the dataset.

Analyzed image sizes: 64, 128, 256, 512, 1024

## Performance Across Image Sizes

### Cardiomegaly

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 64 | 0.7592 | 0.8310 | 0.7631 | 0.8475 | 0.8031 |
| 128 | 0.7716 | 0.8420 | 0.7702 | 0.8636 | 0.8143 |
| 256 | 0.7926 | 0.8662 | 0.7934 | 0.8681 | 0.8291 |
| 512 | 0.8109 | 0.8887 | 0.8125 | 0.8757 | 0.8429 |
| 1024 | 0.8222 | 0.8991 | 0.8492 | 0.8429 | 0.8460 |

**Best image size for Cardiomegaly by F1 score**: 1024 (F1: 0.8460)

### Lung Opacity

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 64 | 0.8156 | 0.8665 | 0.8481 | 0.9058 | 0.8760 |
| 128 | 0.8320 | 0.8830 | 0.8563 | 0.9210 | 0.8874 |
| 256 | 0.8503 | 0.9089 | 0.8860 | 0.9089 | 0.8973 |
| 512 | 0.8679 | 0.9287 | 0.9000 | 0.9183 | 0.9091 |
| 1024 | 0.8758 | 0.9357 | 0.9046 | 0.9247 | 0.9146 |

**Best image size for Lung Opacity by F1 score**: 1024 (F1: 0.9146)

### Edema

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 64 | 0.7413 | 0.8207 | 0.7062 | 0.7575 | 0.7309 |
| 128 | 0.7510 | 0.8315 | 0.7090 | 0.7859 | 0.7455 |
| 256 | 0.7728 | 0.8567 | 0.7344 | 0.7994 | 0.7655 |
| 512 | 0.7981 | 0.8863 | 0.7607 | 0.8240 | 0.7911 |
| 1024 | 0.8099 | 0.8969 | 0.8100 | 0.7711 | 0.7901 |

**Best image size for Edema by F1 score**: 512 (F1: 0.7911)

### No Finding

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 64 | 0.8789 | 0.8650 | 0.5603 | 0.3321 | 0.4170 |
| 128 | 0.8894 | 0.8913 | 0.6095 | 0.4231 | 0.4994 |
| 256 | 0.9017 | 0.9213 | 0.6422 | 0.5561 | 0.5960 |
| 512 | 0.9153 | 0.9427 | 0.6861 | 0.6468 | 0.6658 |
| 1024 | 0.9278 | 0.9522 | 0.7419 | 0.6840 | 0.7118 |

**Best image size for No Finding by F1 score**: 1024 (F1: 0.7118)

### Pneumothorax

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 64 | 0.8743 | 0.6887 | 0.4124 | 0.0548 | 0.0968 |
| 128 | 0.8732 | 0.7320 | 0.4581 | 0.1770 | 0.2554 |
| 256 | 0.8804 | 0.7955 | 0.5256 | 0.2651 | 0.3525 |
| 512 | 0.9002 | 0.8680 | 0.6871 | 0.3433 | 0.4578 |
| 1024 | 0.9101 | 0.9007 | 0.6852 | 0.4962 | 0.5756 |

**Best image size for Pneumothorax by F1 score**: 1024 (F1: 0.5756)

### Pleural Effusion

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 64 | 0.7165 | 0.7876 | 0.6880 | 0.7647 | 0.7243 |
| 128 | 0.7448 | 0.8207 | 0.7086 | 0.8086 | 0.7553 |
| 256 | 0.7855 | 0.8638 | 0.7564 | 0.8256 | 0.7895 |
| 512 | 0.8100 | 0.8900 | 0.7916 | 0.8280 | 0.8094 |
| 1024 | 0.8204 | 0.8995 | 0.7953 | 0.8499 | 0.8217 |

**Best image size for Pleural Effusion by F1 score**: 1024 (F1: 0.8217)

## Best Image Size per Disease

| Disease | Best Image Size | F1 Score | AUROC |
| ------- | --------------- | -------- | ----- |
| Cardiomegaly | 1024 | 0.8460 | 0.8991 |
| Lung Opacity | 1024 | 0.9146 | 0.9357 |
| Edema | 512 | 0.7911 | 0.8863 |
| No Finding | 1024 | 0.7118 | 0.9522 |
| Pneumothorax | 1024 | 0.5756 | 0.9007 |
| Pleural Effusion | 1024 | 0.8217 | 0.8995 |

## Conclusion

Based on the average F1 score across all diseases, the overall best image size for efficientnet-b0 (pretrained) is **1024**.

However, as shown in the per-disease analysis, the optimal image size varies across different pathologies. Consider using the disease-specific optimal image sizes if focusing on particular conditions.
