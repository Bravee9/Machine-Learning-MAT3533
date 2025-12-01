# Week 10: K-Means and DBSCAN Clustering

**Student:** Bui Quang Chien  
**Course:** Machine Learning - MAT3533

---

## Summary

Implementation of two clustering algorithms: **K-Means** (distance-based clustering) and **DBSCAN** (density-based clustering).

---

## Contents

### Part 1: K-Means
1. Build algorithm from numpy (Gaussian 3-cluster data)
2. Compare with sklearn
3. Apply to MNIST clustering (5000 samples to 10 clusters)
   - Analyze correct/incorrect ratios per digit
   - Display centroids and representative samples

### Part 2: DBSCAN
1. Build DBSCAN class from numpy
2. Test 9 parameter sets (epsilon x MinPts) on input.csv
3. Compare with sklearn
4. Analyze parameter effects
5. Direct comparison K-Means vs DBSCAN

---

## Key Points

### K-Means
- Advantages: Fast, simple, suitable for spherical clusters
- Disadvantages: Requires predefined K, sensitive to outliers

### DBSCAN
- Advantages: Auto-detects cluster count, identifies noise/outliers, handles arbitrary shapes
- Disadvantages: Difficult to tune epsilon and MinPts parameters

| Criteria | K-Means | DBSCAN |
|----------|---------|--------|
| Cluster count | Predefined | Automatic |
| Shape | Spherical | Arbitrary |
| Outliers | Sensitive | Detects |
| Speed | Fast | Slower |

---

## Running the Code

```bash
# Install dependencies
pip install numpy matplotlib scipy scikit-learn pandas idx2numpy

# Run notebook
jupyter notebook kmeans_dbscan.ipynb
```

---

## Results

- **K-Means MNIST**: ~70-80% accuracy
- **DBSCAN**: Effective cluster + noise detection with optimal parameters (epsilon=0.5, MinPts=5)
- Custom implementation matches sklearn results

---

## Files

- `kmeans_dbscan.ipynb` - Main notebook
- `input.csv` - DBSCAN data  
- `data/` - Additional data

---

**Repository:** [Machine-Learning-MAT3533](https://github.com/Bravee9/Machine-Learning-MAT3533)
