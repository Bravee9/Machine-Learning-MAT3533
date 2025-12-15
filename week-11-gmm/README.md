# Week 11: Gaussian Mixture Model

**English** | [Tiếng Việt](#tiếng-việt)

## Overview

Probabilistic clustering with Gaussian Mixture Models (GMM) using Expectation-Maximization (EM) algorithm. Models data as mixture of multiple Gaussian distributions for soft clustering.

## Dataset

- **Synthetic 2D**: 3000 points, 3 overlapping Gaussian clusters
- **Iris**: 150 samples, 4 features, 3 species
- **Shopping**: 201 customers, income vs spending score, 5 segments
- Comparison with K-means clustering

## Key Concepts

- Expectation-Maximization (E-step: responsibilities, M-step: parameters)
- Soft clustering with probabilistic membership
- Gaussian ellipses with different covariances
- Model selection: BIC and AIC criteria
- Mahalanobis distance metric

## Implementation

- Custom GMM class from NumPy
- E-step: compute responsibilities using Bayes theorem
- M-step: update means, covariances, weights
- Scikit-learn GaussianMixture for production
- Customer segmentation with business insights

## Results

- Log-likelihood convergence monitoring
- Adjusted Rand Index (ARI) evaluation
- 5 customer segments identified (careful spenders, impulsive buyers, premium customers)
- GMM vs K-means: elliptical vs spherical clusters
- Confusion matrix for Iris classification

## Usage

```bash
jupyter notebook thuc_hanh_tuan_11_BuiQuangChien.ipynb
```

---

## Tiếng Việt

## Tổng quan

Phân cụm xác suất với Mô hình Hỗn hợp Gaussian (GMM) sử dụng thuật toán Expectation-Maximization (EM). Mô hình hóa dữ liệu như hỗn hợp của nhiều phân phối Gaussian cho phân cụm mềm.

## Dữ liệu

- **2D tổng hợp**: 3000 điểm, 3 cụm Gaussian chồng chéo
- **Iris**: 150 mẫu, 4 đặc trưng, 3 loài
- **Mua sắm**: 201 khách hàng, thu nhập vs điểm chi tiêu, 5 phân khúc
- So sánh với phân cụm K-means

## Khái niệm chính

- Expectation-Maximization (E-step: trách nhiệm, M-step: tham số)
- Phân cụm mềm với thành viên xác suất
- Ellipse Gaussian với hiệp phương sai khác nhau
- Lựa chọn mô hình: tiêu chí BIC và AIC
- Chỉ số khoảng cách Mahalanobis

## Triển khai

- Lớp GMM tùy chỉnh từ NumPy
- E-step: tính trách nhiệm sử dụng định lý Bayes
- M-step: cập nhật trung bình, hiệp phương sai, trọng số
- Scikit-learn GaussianMixture cho sản xuất
- Phân khúc khách hàng với thông tin kinh doanh

## Kết quả

- Giám sát hội tụ log-likelihood
- Đánh giá Adjusted Rand Index (ARI)
- Xác định 5 phân khúc khách hàng (người chi tiêu cẩn thận, người mua bốc đồng, khách hàng cao cấp)
- GMM vs K-means: cụm ellipse vs hình cầu
- Confusion matrix cho phân loại Iris

## Sử dụng

```bash
jupyter notebook thuc_hanh_tuan_11_BuiQuangChien.ipynb
```
