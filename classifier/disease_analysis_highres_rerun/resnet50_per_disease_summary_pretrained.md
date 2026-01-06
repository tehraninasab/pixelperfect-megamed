# Per-Disease Performance Analysis for resnet50 (Pretrained)

## Overview

This report analyzes the performance of the CheXpert classifier using the resnet50 architecture (pretrained) across different image sizes for each disease in the dataset.

Analyzed image sizes: 64, 128, 256, 512, 1024

## Performance Across Image Sizes

### Cardiomegaly

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 64 | 0.7866 | 0.8535 | 0.8060 | 0.8322 | 0.8189 |
| 128 | 0.7993 | 0.8664 | 0.8163 | 0.8434 | 0.8297 |
| 256 | 0.8122 | 0.8826 | 0.8275 | 0.8540 | 0.8406 |
| 512 | 0.8119 | 0.8879 | 0.8222 | 0.8619 | 0.8416 |
| 1024 | 0.8079 | 0.8857 | 0.8172 | 0.8611 | 0.8386 |

**Best image size for Cardiomegaly by F1 score**: 512 (F1: 0.8416)

### Lung Opacity

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 64 | 0.8497 | 0.9000 | 0.8820 | 0.9132 | 0.8973 |
| 128 | 0.8643 | 0.9109 | 0.8903 | 0.9253 | 0.9075 |
| 256 | 0.8715 | 0.9270 | 0.9026 | 0.9206 | 0.9115 |
| 512 | 0.8707 | 0.9286 | 0.8989 | 0.9241 | 0.9113 |
| 1024 | 0.8691 | 0.9304 | 0.9048 | 0.9141 | 0.9094 |

**Best image size for Lung Opacity by F1 score**: 256 (F1: 0.9115)

### Edema

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 64 | 0.7649 | 0.8379 | 0.7489 | 0.7420 | 0.7454 |
| 128 | 0.7798 | 0.8557 | 0.7671 | 0.7546 | 0.7608 |
| 256 | 0.7959 | 0.8762 | 0.7808 | 0.7787 | 0.7797 |
| 512 | 0.7976 | 0.8821 | 0.7816 | 0.7824 | 0.7820 |
| 1024 | 0.7974 | 0.8816 | 0.7846 | 0.7765 | 0.7805 |

**Best image size for Edema by F1 score**: 512 (F1: 0.7820)

### No Finding

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 64 | 0.9032 | 0.9023 | 0.6785 | 0.4897 | 0.5689 |
| 128 | 0.9113 | 0.9130 | 0.7128 | 0.5360 | 0.6119 |
| 256 | 0.9222 | 0.9405 | 0.7405 | 0.6216 | 0.6759 |
| 512 | 0.9265 | 0.9470 | 0.7699 | 0.6224 | 0.6884 |
| 1024 | 0.9287 | 0.9510 | 0.7668 | 0.6511 | 0.7042 |

**Best image size for No Finding by F1 score**: 1024 (F1: 0.7042)

### Pneumothorax

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 64 | 0.8703 | 0.7442 | 0.4524 | 0.2671 | 0.3359 |
| 128 | 0.8839 | 0.7986 | 0.5419 | 0.3510 | 0.4260 |
| 256 | 0.9033 | 0.8604 | 0.6594 | 0.4396 | 0.5275 |
| 512 | 0.9119 | 0.8877 | 0.6965 | 0.5016 | 0.5832 |
| 1024 | 0.9141 | 0.8951 | 0.7207 | 0.4913 | 0.5843 |

**Best image size for Pneumothorax by F1 score**: 1024 (F1: 0.5843)

### Pleural Effusion

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 64 | 0.7653 | 0.8350 | 0.7585 | 0.7601 | 0.7593 |
| 128 | 0.7878 | 0.8572 | 0.7818 | 0.7830 | 0.7824 |
| 256 | 0.8072 | 0.8802 | 0.8078 | 0.7928 | 0.8003 |
| 512 | 0.8025 | 0.8783 | 0.7984 | 0.7955 | 0.7969 |
| 1024 | 0.8002 | 0.8815 | 0.8028 | 0.7817 | 0.7921 |

**Best image size for Pleural Effusion by F1 score**: 256 (F1: 0.8003)

## Best Image Size per Disease

| Disease | Best Image Size | F1 Score | AUROC |
| ------- | --------------- | -------- | ----- |
| Cardiomegaly | 512 | 0.8416 | 0.8879 |
| Lung Opacity | 256 | 0.9115 | 0.9270 |
| Edema | 512 | 0.7820 | 0.8821 |
| No Finding | 1024 | 0.7042 | 0.9510 |
| Pneumothorax | 1024 | 0.5843 | 0.8951 |
| Pleural Effusion | 256 | 0.8003 | 0.8802 |

## Conclusion

Based on the average F1 score across all diseases, the overall best image size for resnet50 (pretrained) is **1024**.

However, as shown in the per-disease analysis, the optimal image size varies across different pathologies. Consider using the disease-specific optimal image sizes if focusing on particular conditions.
