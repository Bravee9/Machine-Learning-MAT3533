# Week 10: K-Means and DBSCAN Clustering

**English** | [Tiếng Việt](#tiếng-việt)

## Overview

Two unsupervised clustering algorithms: K-Means for distance-based partitioning and DBSCAN for density-based clustering with automatic cluster detection.

## Dataset

- **Gaussian 3-cluster**: Synthetic 2D data for K-Means
- **MNIST**: 5000 samples clustered to 10 digit groups
- **Input.csv**: DBSCAN parameter testing data
- Custom implementation vs scikit-learn comparison

## Key Concepts

- K-Means: centroid-based, spherical clusters, predefined k
- DBSCAN: density-based, arbitrary shapes, auto cluster count
- Distance metrics and density parameters
- Noise and outlier detection
- Cluster evaluation metrics

## Implementation

- K-Means from NumPy: Gaussian and MNIST clustering
- DBSCAN with epsilon and MinPts parameter tuning
- Centroid visualization and representative samples
- 9 parameter sets comparison for DBSCAN

## Results

- K-Means MNIST: ~70-80% accuracy
- DBSCAN: effective noise detection with optimal parameters (ε=0.5, MinPts=5)
- Custom implementation matches scikit-learn
- K-Means vs DBSCAN comparative analysis

## Usage

```bash
jupyter notebook kmeans_dbscan.ipynb
```

---

## Tiếng Việt

## Tổng quan

Hai thuật toán phân cụm không giám sát: K-Means cho phân vùng dựa trên khoảng cách và DBSCAN cho phân cụm dựa trên mật độ với phát hiện cụm tự động.

## Dữ liệu

- **Gaussian 3 cụm**: Dữ liệu 2D tổng hợp cho K-Means
- **MNIST**: 5000 mẫu được phân cụm thành 10 nhóm chữ số
- **Input.csv**: Dữ liệu kiểm tra tham số DBSCAN
- So sánh cài đặt tùy chỉnh vs scikit-learn

## Khái niệm chính

- K-Means: dựa trên tâm, cụm hình cầu, k định trước
- DBSCAN: dựa trên mật độ, hình dạng tùy ý, số cụm tự động
- Chỉ số khoảng cách và tham số mật độ
- Phát hiện nhiễu và ngoại lai
- Các chỉ số đánh giá cụm

## Triển khai

- K-Means từ NumPy: phân cụm Gaussian và MNIST
- DBSCAN với điều chỉnh tham số epsilon và MinPts
- Trực quan hóa tâm cụm và mẫu đại diện
- So sánh 9 bộ tham số cho DBSCAN

## Kết quả

- K-Means MNIST: độ chính xác ~70-80%
- DBSCAN: phát hiện nhiễu hiệu quả với tham số tối ưu (ε=0.5, MinPts=5)
- Cài đặt tùy chỉnh khớp với scikit-learn
- Phân tích so sánh K-Means vs DBSCAN

## Sử dụng

```bash
jupyter notebook kmeans_dbscan.ipynb
```
