# Week 1: Gaussian Naive Bayes

## Algorithm

Gaussian Naive Bayes classifier for email spam detection. Implements probabilistic classification based on Bayes' theorem with the assumption of feature independence and Gaussian distribution.

## Dataset

**Email Spam Classification**
- 4601 total samples
- Multiple training sizes: 50, 100, 400, full dataset
- Binary classification: spam vs non-spam
- Features: word frequencies and character statistics

Files:
- `train-features.txt` - Full training features
- `train-labels.txt` - Training labels
- `test-features.txt` - Test features
- `test-labels.txt` - Test labels
- Subsets: 50, 100, 400 sample versions

## Implementation

### Key Concepts
- Bayes' Theorem application
- Gaussian probability density function
- Maximum likelihood estimation
- Feature independence assumption

### Methods
- Manual implementation from scratch
- Scikit-learn GaussianNB comparison
- Training with different dataset sizes

## Results

Performance metrics across different training sizes:
- Effect of training set size on accuracy
- Precision and recall analysis
- Confusion matrix visualization

## Usage

```python
jupyter notebook gaussian_nb.ipynb
```

## Key Learnings

- Understanding probabilistic classification
- Impact of training data size on model performance
- Naive Bayes assumptions and limitations
- Feature independence in real-world datasets
