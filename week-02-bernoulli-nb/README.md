# Week 2: Bernoulli Naive Bayes

## Algorithm

Bernoulli Naive Bayes for binary feature classification. Designed for binary/boolean features, particularly effective for text classification with binary word occurrence features.

## Dataset

**Medical Diagnosis - Symptom-based Disease Prediction**
- Binary features representing presence/absence of symptoms
- Multiple disease categories
- Medical symptom patterns

File:
- `bernoulli_nb_symptoms.csv` - Symptom and diagnosis data

## Implementation

### Key Concepts
- Bernoulli distribution for binary features
- Binary feature encoding
- Likelihood calculation for binary data
- Prior probability estimation

### Methods
- Data preprocessing for binary features
- Model training with Bernoulli NB
- Comparison with Gaussian NB
- Feature importance analysis

## Results

- Classification accuracy on medical diagnoses
- Performance comparison with other NB variants
- Symptom pattern analysis
- Confusion matrix and classification report

## Usage

```python
jupyter notebook bernoulli_nb.ipynb
```

## Key Learnings

- Appropriate algorithm selection for data types
- Binary feature representation
- Medical diagnosis prediction challenges
- Bernoulli vs Gaussian NB trade-offs
