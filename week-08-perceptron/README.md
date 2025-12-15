# Week 8: Perceptron

**English** | [Tiếng Việt](#tiếng-việt)

## Overview

Foundation of neural networks: the Perceptron algorithm for binary linear classification. Implements online learning with weight update rules.

## Dataset

- **Sonar**: 208 samples, Rock vs Mine classification, 60 frequency features
- **Portfolio Analysis**: Financial task classification, binary outcomes
- Linearly separable and challenging decision boundaries

## Key Concepts

- Linear binary classifier: f(x) = sign(w · x + b)
- Weight update rule: w = w + η(y - ŷ)x
- Learning rate selection
- Perceptron convergence theorem
- Linear separability requirement

## Implementation

- Weight initialization strategies
- Epoch-based training with convergence criteria
- Decision boundary visualization
- Performance monitoring per epoch
- Bias term importance

## Results

- Convergence behavior over epochs
- Classification accuracy and error rate
- Weight vector interpretation
- Decision boundary geometry
- Training time analysis

## Usage

```bash
jupyter notebook perceptron.ipynb
```

---

## Tiếng Việt

## Tổng quan

Nền tảng của mạng nơ-ron: thuật toán Perceptron cho phân loại tuyến tính nhị phân. Triển khai học trực tuyến với quy tắc cập nhật trọng số.

## Dữ liệu

- **Sonar**: 208 mẫu, phân loại Đá vs Mìn, 60 đặc trưng tần số
- **Phân tích danh mục**: Phân loại nhiệm vụ tài chính, kết quả nhị phân
- Ranh giới quyết định có thể phân tách tuyến tính và khó

## Khái niệm chính

- Phân loại nhị phân tuyến tính: f(x) = sign(w · x + b)
- Quy tắc cập nhật trọng số: w = w + η(y - ŷ)x
- Lựa chọn tốc độ học
- Định lý hội tụ Perceptron
- Yêu cầu phân tách tuyến tính

## Triển khai

- Chiến lược khởi tạo trọng số
- Huấn luyện theo epoch với tiêu chí hội tụ
- Trực quan hóa ranh giới quyết định
- Giám sát hiệu suất theo epoch
- Tầm quan trọng của bias

## Kết quả

- Hành vi hội tụ qua các epoch
- Độ chính xác phân loại và tỷ lệ lỗi
- Giải thích vector trọng số
- Hình học ranh giới quyết định
- Phân tích thời gian huấn luyện

## Sử dụng

```bash
jupyter notebook perceptron.ipynb
```
