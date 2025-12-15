# Week 12: Support Vector Machine (Hard Margin)

**English** | [Tiếng Việt](#tiếng-việt)

## Overview

Hard Margin Support Vector Machine (SVM) for binary classification on linearly separable data. Maximizes margin between classes using quadratic programming optimization.

## Dataset

- **Random 2D**: 20 points, 2 classes, linearly separable Gaussian data
- **Wisconsin Breast Cancer**: 569 samples, 30 features, Benign vs Malignant
- **Sonar**: 208 samples, 60 frequency features, Rock vs Mine
- CVXOPT vs scikit-learn comparison

## Key Concepts

- Maximum margin hyperplane
- Support vectors: points defining decision boundary
- Quadratic programming (primal and dual problems)
- Hard margin: linearly separable data only
- Hyperparameter C (large C for hard margin)

## Implementation

- CVXOPT for direct quadratic programming solution
- Scikit-learn SVC with linear kernel
- Weight and bias calculation from Lagrange multipliers
- Decision boundary visualization

## Results

- Random 2D: 100% accuracy (linearly separable)
- Breast Cancer: ~95%+ accuracy
- Sonar: ~70-85% accuracy (more complex)
- Custom CVXOPT matches scikit-learn results

## Usage

```bash
jupyter notebook BuiQuangChien_23001837.ipynb
```

---

## Tiếng Việt

## Tổng quan

Máy Vector Hỗ trợ Biên Cứng (Hard Margin SVM) cho phân loại nhị phân trên dữ liệu phân tách tuyến tính. Tối đa hóa biên giữa các lớp sử dụng tối ưu hóa quy hoạch bậc hai.

## Dữ liệu

- **2D ngẫu nhiên**: 20 điểm, 2 lớp, dữ liệu Gaussian phân tách tuyến tính
- **Ung thư vú Wisconsin**: 569 mẫu, 30 đặc trưng, Lành tính vs Ác tính
- **Sonar**: 208 mẫu, 60 đặc trưng tần số, Đá vs Mìn
- So sánh CVXOPT vs scikit-learn

## Khái niệm chính

- Siêu phẳng biên tối đa
- Support vectors: điểm xác định ranh giới quyết định
- Quy hoạch bậc hai (bài toán primal và dual)
- Hard margin: chỉ cho dữ liệu phân tách tuyến tính
- Siêu tham số C (C lớn cho hard margin)

## Triển khai

- CVXOPT cho giải pháp quy hoạch bậc hai trực tiếp
- Scikit-learn SVC với kernel tuyến tính
- Tính toán trọng số và bias từ nhân tử Lagrange
- Trực quan hóa ranh giới quyết định

## Kết quả

- 2D ngẫu nhiên: độ chính xác 100% (phân tách tuyến tính)
- Ung thư vú: độ chính xác ~95%+
- Sonar: độ chính xác ~70-85% (phức tạp hơn)
- CVXOPT tùy chỉnh khớp với kết quả scikit-learn

## Sử dụng

```bash
jupyter notebook BuiQuangChien_23001837.ipynb
```
