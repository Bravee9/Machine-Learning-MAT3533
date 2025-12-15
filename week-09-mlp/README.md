# Week 9: Multi-Layer Perceptron

**English** | [Tiếng Việt](#tiếng-việt)

## Overview

Deep learning fundamentals with Multi-Layer Perceptron (MLP) neural networks. Implements backpropagation for both classification and regression tasks.

## Dataset

- **Dry Bean**: 13,611 samples, 16 geometric features, 7 bean types
- **SAT-GPA**: 84 students, SAT score to GPA regression
- Multi-class classification and continuous prediction

## Key Concepts

- Forward and backpropagation algorithms
- Activation functions (ReLU, Tanh, Sigmoid, Softmax)
- Gradient descent optimization (Adam, SGD)
- Early stopping and L2 regularization
- Universal approximation theorem

## Implementation

- Classification: 16-100-50-7 architecture, cross-entropy loss
- Regression: 1-50-25-1 architecture, MSE loss
- Hyperparameter tuning and model comparison
- Adaptive learning rate with Adam optimizer

## Results

- Test accuracy across network configurations
- Confusion matrix for 7-class classification
- R², MSE, RMSE, MAE for regression
- Training loss convergence curves
- Comparison with traditional ML algorithms

## Usage

```bash
jupyter notebook mlp.ipynb
```

---

## Tiếng Việt

## Tổng quan

Các nguyên tắc cơ bản của deep learning với mạng nơ-ron Perceptron Đa tầng (MLP). Triển khai lan truyền ngược cho cả phân loại và hồi quy.

## Dữ liệu

- **Đậu khô**: 13,611 mẫu, 16 đặc trưng hình học, 7 loại đậu
- **SAT-GPA**: 84 sinh viên, hồi quy điểm SAT sang GPA
- Phân loại đa lớp và dự đoán liên tục

## Khái niệm chính

- Thuật toán truyền thẳng và lan truyền ngược
- Hàm kích hoạt (ReLU, Tanh, Sigmoid, Softmax)
- Tối ưu hóa gradient descent (Adam, SGD)
- Dừng sớm và regularization L2
- Định lý xấp xỉ vũ trụ

## Triển khai

- Phân loại: kiến trúc 16-100-50-7, cross-entropy loss
- Hồi quy: kiến trúc 1-50-25-1, MSE loss
- Điều chỉnh siêu tham số và so sánh mô hình
- Tốc độ học thích ứng với Adam optimizer

## Kết quả

- Độ chính xác kiểm tra qua các cấu hình mạng
- Confusion matrix cho phân loại 7 lớp
- R², MSE, RMSE, MAE cho hồi quy
- Đường cong hội tụ loss huấn luyện
- So sánh với các thuật toán ML truyền thống

## Sử dụng

```bash
jupyter notebook mlp.ipynb
```
