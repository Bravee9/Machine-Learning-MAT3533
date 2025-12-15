# Week 7: Linear Discriminant Analysis

**English** | [Tiếng Việt](#tiếng-việt)

## Overview

Supervised dimensionality reduction using Linear Discriminant Analysis (LDA). Maximizes class separation for improved classification on MNIST and face recognition tasks.

## Dataset

- **MNIST**: 70,000 samples (28x28 images), 784 features, 10 classes
- **Face Recognition**: Multiple subjects, high-dimensional image data
- Reduction to k-1 discriminant components (9 for 10 classes)

## Key Concepts

- Fisher's linear discriminant
- Between-class and within-class scatter
- Generalized eigenvalue problem
- Maximum class separability
- Supervised vs unsupervised reduction

## Implementation

- Feature standardization
- Scatter matrix computation (S_W, S_B)
- Eigenvalue problem solving
- Dimensionality reduction: 784 to 9 components
- Classification with reduced features

## Results

- Classification accuracy before/after LDA
- Confusion matrix analysis
- Training time and memory efficiency
- 2D/3D class separation visualization
- Comparison with PCA performance

## Usage

```bash
jupyter notebook lda.ipynb
```

---

## Tiếng Việt

## Tổng quan

Giảm chiều có giám sát sử dụng Phân tích Phân biệt Tuyến tính (LDA). Tối đa hóa sự phân tách lớp để cải thiện phân loại trên MNIST và nhận dạng khuôn mặt.

## Dữ liệu

- **MNIST**: 70,000 mẫu (ảnh 28x28), 784 đặc trưng, 10 lớp
- **Nhận dạng khuôn mặt**: Nhiều đối tượng, dữ liệu ảnh đa chiều
- Giảm về k-1 thành phần phân biệt (9 cho 10 lớp)

## Khái niệm chính

- Phân biệt tuyến tính Fisher
- Độ phân tán giữa lớp và trong lớp
- Bài toán trị riêng tổng quát
- Khả năng phân tách lớp tối đa
- Giảm chiều có giám sát vs không giám sát

## Triển khai

- Chuẩn hóa đặc trưng
- Tính toán ma trận phân tán (S_W, S_B)
- Giải bài toán trị riêng
- Giảm chiều: 784 về 9 thành phần
- Phân loại với đặc trưng đã giảm

## Kết quả

- Độ chính xác phân loại trước/sau LDA
- Phân tích confusion matrix
- Thời gian huấn luyện và hiệu quả bộ nhớ
- Trực quan hóa phân tách lớp 2D/3D
- So sánh hiệu suất với PCA

## Sử dụng

```bash
jupyter notebook lda.ipynb
```
