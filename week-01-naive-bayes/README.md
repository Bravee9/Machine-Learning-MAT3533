# Week 1: Gaussian Naive Bayes

**English** | [Tiếng Việt](#tiếng-việt)

## Overview

Probabilistic classification using Gaussian Naive Bayes for email spam detection. Implements Bayes' theorem with feature independence assumption and Gaussian distribution.

## Dataset

- **Email Spam**: 4601 samples, binary classification (spam/non-spam)
- Training subsets: 50, 100, 400, full dataset
- Features: word frequencies and character statistics

## Key Concepts

- Bayes' theorem and conditional probability
- Gaussian probability density function
- Maximum likelihood estimation
- Feature independence assumption

## Implementation

- Manual implementation from scratch
- Comparison with scikit-learn GaussianNB
- Training with various dataset sizes

## Results

- Accuracy improves with training set size
- High performance on binary classification
- Confusion matrix and precision/recall analysis

## Usage

```bash
jupyter notebook gaussian_nb.ipynb
```

---

## Tiếng Việt

## Tổng quan

Phân loại xác suất sử dụng Gaussian Naive Bayes cho phát hiện email spam. Áp dụng định lý Bayes với giả định độc lập đặc trưng và phân phối Gaussian.

## Dữ liệu

- **Email Spam**: 4601 mẫu, phân loại nhị phân (spam/không spam)
- Tập con huấn luyện: 50, 100, 400, toàn bộ dữ liệu
- Đặc trưng: tần suất từ và thống kê ký tự

## Khái niệm chính

- Định lý Bayes và xác suất có điều kiện
- Hàm mật độ xác suất Gaussian
- Ước lượng hợp lý cực đại
- Giả định độc lập đặc trưng

## Triển khai

- Cài đặt thủ công từ đầu
- So sánh với scikit-learn GaussianNB
- Huấn luyện với nhiều kích thước dữ liệu

## Kết quả

- Độ chính xác tăng theo kích thước tập huấn luyện
- Hiệu suất cao trong phân loại nhị phân
- Phân tích confusion matrix và precision/recall

## Sử dụng

```bash
jupyter notebook gaussian_nb.ipynb
```
