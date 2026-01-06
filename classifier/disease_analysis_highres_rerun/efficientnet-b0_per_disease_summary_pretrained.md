# Per-Disease Performance Analysis for efficientnet-b0 (Pretrained)

## Overview

This report analyzes the performance of the CheXpert classifier using the efficientnet-b0 architecture (pretrained) across different image sizes for each disease in the dataset.

Analyzed image sizes: 64, 128, 256, 512, 1024

## Performance Across Image Sizes

### Cardiomegaly

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 64 | 0.7887 | 0.8641 | 0.8042 | 0.8399 | 0.8217 |
| 128 | 0.8080 | 0.8795 | 0.8146 | 0.8658 | 0.8394 |
| 256 | 0.8139 | 0.8879 | 0.8243 | 0.8629 | 0.8431 |
| 512 | 0.8168 | 0.8900 | 0.8310 | 0.8585 | 0.8445 |
| 1024 | 0.8174 | 0.8917 | 0.8316 | 0.8588 | 0.8450 |

**Best image size for Cardiomegaly by F1 score**: 1024 (F1: 0.8450)

### Lung Opacity

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 64 | 0.8514 | 0.9090 | 0.8911 | 0.9038 | 0.8974 |
| 128 | 0.8663 | 0.9242 | 0.8961 | 0.9210 | 0.9083 |
| 256 | 0.8712 | 0.9297 | 0.9040 | 0.9185 | 0.9112 |
| 512 | 0.8729 | 0.9304 | 0.9078 | 0.9163 | 0.9120 |
| 1024 | 0.8759 | 0.9330 | 0.9025 | 0.9277 | 0.9149 |

**Best image size for Lung Opacity by F1 score**: 1024 (F1: 0.9149)

### Edema

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 64 | 0.7776 | 0.8563 | 0.7555 | 0.7696 | 0.7625 |
| 128 | 0.7952 | 0.8789 | 0.7681 | 0.8003 | 0.7838 |
| 256 | 0.8045 | 0.8890 | 0.7853 | 0.7962 | 0.7907 |
| 512 | 0.8058 | 0.8904 | 0.7893 | 0.7930 | 0.7912 |
| 1024 | 0.8091 | 0.8946 | 0.7983 | 0.7875 | 0.7929 |

**Best image size for Edema by F1 score**: 1024 (F1: 0.7929)

### No Finding

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 64 | 0.9005 | 0.9173 | 0.6336 | 0.5622 | 0.5958 |
| 128 | 0.9156 | 0.9359 | 0.7105 | 0.5957 | 0.6480 |
| 256 | 0.9213 | 0.9461 | 0.7097 | 0.6711 | 0.6899 |
| 512 | 0.9272 | 0.9498 | 0.7320 | 0.6965 | 0.7138 |
| 1024 | 0.9308 | 0.9531 | 0.7626 | 0.6818 | 0.7200 |

**Best image size for No Finding by F1 score**: 1024 (F1: 0.7200)

### Pneumothorax

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 64 | 0.8788 | 0.7713 | 0.5176 | 0.1884 | 0.2762 |
| 128 | 0.8934 | 0.8420 | 0.6257 | 0.3296 | 0.4318 |
| 256 | 0.9028 | 0.8798 | 0.6515 | 0.4490 | 0.5316 |
| 512 | 0.9141 | 0.9099 | 0.7061 | 0.5155 | 0.5959 |
| 1024 | 0.9170 | 0.9168 | 0.7146 | 0.5394 | 0.6147 |

**Best image size for Pneumothorax by F1 score**: 1024 (F1: 0.6147)

### Pleural Effusion

| Image Size | Accuracy | AUROC | Precision | Recall | F1 |
| ---------- | -------- | ----- | --------- | ------ | -- |
| 64 | 0.7756 | 0.8565 | 0.7575 | 0.7933 | 0.7750 |
| 128 | 0.8025 | 0.8822 | 0.7877 | 0.8138 | 0.8006 |
| 256 | 0.8130 | 0.8937 | 0.7956 | 0.8291 | 0.8120 |
| 512 | 0.8186 | 0.8951 | 0.8034 | 0.8310 | 0.8170 |
| 1024 | 0.8184 | 0.8964 | 0.7908 | 0.8528 | 0.8206 |

**Best image size for Pleural Effusion by F1 score**: 1024 (F1: 0.8206)

## Best Image Size per Disease

| Disease | Best Image Size | F1 Score | AUROC |
| ------- | --------------- | -------- | ----- |
| Cardiomegaly | 1024 | 0.8450 | 0.8917 |
| Lung Opacity | 1024 | 0.9149 | 0.9330 |
| Edema | 1024 | 0.7929 | 0.8946 |
| No Finding | 1024 | 0.7200 | 0.9531 |
| Pneumothorax | 1024 | 0.6147 | 0.9168 |
| Pleural Effusion | 1024 | 0.8206 | 0.8964 |

## Conclusion

Based on the average F1 score across all diseases, the overall best image size for efficientnet-b0 (pretrained) is **1024**.

However, as shown in the per-disease analysis, the optimal image size varies across different pathologies. Consider using the disease-specific optimal image sizes if focusing on particular conditions.
