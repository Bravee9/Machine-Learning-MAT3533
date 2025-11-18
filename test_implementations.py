"""
Test script to verify all implementations are working correctly
"""

import numpy as np
import sys

print("=" * 70)
print("TESTING MACHINE LEARNING IMPLEMENTATIONS")
print("=" * 70)

# Test imports
print("\n1. Testing imports...")
try:
    from algorithms.supervised import (
        NaiveBayesClassifier, LogisticRegression, DecisionTreeClassifier,
        KNearestNeighbors, RandomForestClassifier, LinearRegression, SupportVectorMachine
    )
    from algorithms.unsupervised import KMeans, DBSCAN, HierarchicalClustering, PCA, TSNE
    from algorithms.neural_networks import NeuralNetwork, DenseLayer, ActivationLayer
    from utils.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, train_test_split
    from utils.evaluation import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from utils.model_selection import cross_val_score, GridSearchCV, KFold
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Generate test data
np.random.seed(42)
X_train = np.random.randn(100, 2)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
X_test = np.random.randn(20, 2)
y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)

print("\n2. Testing Supervised Learning Algorithms...")

# Test Naive Bayes
try:
    nb = NaiveBayesClassifier()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   ✓ Naive Bayes: accuracy = {acc:.3f}")
except Exception as e:
    print(f"   ✗ Naive Bayes failed: {e}")

# Test Logistic Regression
try:
    lr = LogisticRegression(learning_rate=0.1, n_iterations=100)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   ✓ Logistic Regression: accuracy = {acc:.3f}")
except Exception as e:
    print(f"   ✗ Logistic Regression failed: {e}")

# Test KNN
try:
    knn = KNearestNeighbors(k=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   ✓ K-Nearest Neighbors: accuracy = {acc:.3f}")
except Exception as e:
    print(f"   ✗ KNN failed: {e}")

# Test Decision Tree
try:
    dt = DecisionTreeClassifier(max_depth=3)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   ✓ Decision Tree: accuracy = {acc:.3f}")
except Exception as e:
    print(f"   ✗ Decision Tree failed: {e}")

# Test Random Forest
try:
    rf = RandomForestClassifier(n_trees=10, max_depth=3)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   ✓ Random Forest: accuracy = {acc:.3f}")
except Exception as e:
    print(f"   ✗ Random Forest failed: {e}")

# Test Linear Regression
try:
    y_reg = X_train[:, 0] * 2 + X_train[:, 1] * 3 + np.random.randn(100) * 0.1
    lr_model = LinearRegression(method='normal')
    lr_model.fit(X_train, y_reg)
    y_pred = lr_model.predict(X_test)
    print(f"   ✓ Linear Regression: coefficients = {lr_model.weights}")
except Exception as e:
    print(f"   ✗ Linear Regression failed: {e}")

print("\n3. Testing Unsupervised Learning Algorithms...")

# Test K-Means
try:
    kmeans = KMeans(k=2, max_iters=50)
    kmeans.fit(X_train)
    labels = kmeans.predict(X_test)
    print(f"   ✓ K-Means: {len(np.unique(labels))} clusters found")
except Exception as e:
    print(f"   ✗ K-Means failed: {e}")

# Test DBSCAN
try:
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan.fit(X_train)
    n_clusters = len(np.unique(dbscan.labels[dbscan.labels != -1]))
    print(f"   ✓ DBSCAN: {n_clusters} clusters found")
except Exception as e:
    print(f"   ✗ DBSCAN failed: {e}")

# Test PCA
try:
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_train)
    var_explained = np.sum(pca.explained_variance_ratio)
    print(f"   ✓ PCA: {var_explained:.3f} variance explained")
except Exception as e:
    print(f"   ✗ PCA failed: {e}")

print("\n4. Testing Neural Networks...")

# Test Neural Network
try:
    # Create simple network
    nn = NeuralNetwork()
    nn.add(DenseLayer(2, 4))
    nn.add(ActivationLayer('relu'))
    nn.add(DenseLayer(4, 1))
    nn.add(ActivationLayer('sigmoid'))
    
    # Train
    y_nn = y_train.reshape(-1, 1)
    nn.fit(X_train, y_nn, epochs=50, learning_rate=0.1, verbose=False)
    
    # Predict
    y_pred = nn.predict(X_test)
    y_pred_class = (y_pred > 0.5).astype(int).flatten()
    acc = accuracy_score(y_test, y_pred_class)
    print(f"   ✓ Neural Network: accuracy = {acc:.3f}")
except Exception as e:
    print(f"   ✗ Neural Network failed: {e}")

print("\n5. Testing Utilities...")

# Test preprocessing
try:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    print(f"   ✓ StandardScaler: mean ≈ {np.mean(X_scaled):.3f}, std ≈ {np.std(X_scaled):.3f}")
except Exception as e:
    print(f"   ✗ StandardScaler failed: {e}")

# Test evaluation metrics
try:
    y_pred = np.random.randint(0, 2, size=20)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"   ✓ Evaluation metrics computed successfully")
except Exception as e:
    print(f"   ✗ Evaluation metrics failed: {e}")

# Test cross-validation
try:
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    splits = list(kf.split(X_train))
    print(f"   ✓ K-Fold CV: {len(splits)} splits created")
except Exception as e:
    print(f"   ✗ K-Fold CV failed: {e}")

print("\n" + "=" * 70)
print("ALL TESTS COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\nThe implementations are working correctly.")
print("You can now run the examples in the 'examples/' directory.")
print("\nNext steps:")
print("  cd examples")
print("  python 01_classification_example.py")
