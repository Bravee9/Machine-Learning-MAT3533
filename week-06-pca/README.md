# Week 6: Principal Component Analysis

**English** | [Tiếng Việt](#tiếng-việt)

## Overview

Dimensionality reduction using Principal Component Analysis (PCA) for high-dimensional speech feature data. Transforms data to maximize variance in reduced dimensions.

## Dataset

- **Parkinson's Disease Speech**: 756 voice recordings
- Original features: 754 speech characteristics
- Classes: Parkinson's patient vs healthy control
- High dimensionality challenge

## Key Concepts

- Eigenvalue and eigenvector computation
- Variance maximization
- Covariance matrix analysis
- Linear transformation
- Curse of dimensionality mitigation

## Implementation

- Feature standardization (zero mean, unit variance)
- Eigenvalue decomposition
- Scree plot for component selection
- Cumulative explained variance analysis
- Dimensionality reduction to optimal k components

## Results

- Explained variance ratio per component
- Cumulative variance preserved
- Classification performance before/after PCA
- 2D/3D visualization in principal component space
- Noise filtering and computational cost reduction

## Usage

```bash
jupyter notebook pca.ipynb
```

---

## Tiếng Việt

## Tổng quan

Giảm chiều dữ liệu sử dụng Phân tích Thành phần Chính (PCA) cho dữ liệu đặc trưng giọng nói đa chiều. Biến đổi dữ liệu để tối đa hóa phương sai trong không gian chiều thấp hơn.

## Dữ liệu

- **Giọng nói bệnh Parkinson**: 756 bản ghi âm
- Đặc trưng gốc: 754 đặc điểm giọng nói
- Lớp: Bệnh nhân Parkinson vs người khỏe mạnh
- Thách thức đa chiều cao

## Khái niệm chính

- Tính toán trị riêng và vector riêng
- Tối đa hóa phương sai
- Phân tích ma trận hiệp phương sai
- Biến đổi tuyến tính
- Giảm thiểu lời nguyền chiều cao

## Triển khai

- Chuẩn hóa đặc trưng (trung bình 0, phương sai 1)
- Phân tích trị riêng
- Biểu đồ Scree để chọn thành phần
- Phân tích phương sai tích lũy
- Giảm chiều về k thành phần tối ưu

## Kết quả

- Tỷ lệ phương sai giải thích cho mỗi thành phần
- Phương sai tích lũy được bảo toàn
- Hiệu suất phân loại trước/sau PCA
- Trực quan hóa 2D/3D trong không gian thành phần chính
- Lọc nhiễu và giảm chi phí tính toán

## Sử dụng

```bash
jupyter notebook pca.ipynb
```
