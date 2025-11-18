"""
Example 5: Neural Networks
Demonstrates feedforward neural network for classification
"""

import numpy as np
import sys
sys.path.append('..')

from algorithms.neural_networks import NeuralNetwork, DenseLayer, ActivationLayer
from utils.preprocessing import StandardScaler, train_test_split, OneHotEncoder
from utils.evaluation import accuracy_score

# Generate synthetic data for multi-class classification
np.random.seed(42)

# Create 3 classes
n_per_class = 100
X_class0 = np.random.randn(n_per_class, 2) + np.array([-2, -2])
X_class1 = np.random.randn(n_per_class, 2) + np.array([2, -2])
X_class2 = np.random.randn(n_per_class, 2) + np.array([0, 2])

X = np.vstack([X_class0, X_class1, X_class2])
y = np.concatenate([
    np.zeros(n_per_class),
    np.ones(n_per_class),
    np.ones(n_per_class) * 2
]).astype(int)

# Split and preprocess
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One-hot encode labels
encoder = OneHotEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

print("=" * 70)
print("NEURAL NETWORK EXAMPLE - Multi-class Classification")
print("=" * 70)
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Number of features: {X_train.shape[1]}")
print(f"Number of classes: {len(np.unique(y))}")
print()

# Build neural network
print("\nBuilding Neural Network Architecture:")
print("-" * 50)

nn = NeuralNetwork()
nn.add(DenseLayer(input_size=2, output_size=8))
nn.add(ActivationLayer('relu'))
nn.add(DenseLayer(input_size=8, output_size=8))
nn.add(ActivationLayer('relu'))
nn.add(DenseLayer(input_size=8, output_size=3))
nn.add(ActivationLayer('softmax'))

print("Layer 1: Dense (2 -> 8) + ReLU")
print("Layer 2: Dense (8 -> 8) + ReLU")
print("Layer 3: Dense (8 -> 3) + Softmax")
print("\nTotal layers: 6 (3 dense + 3 activation)")

# Train network
print("\nTraining Neural Network...")
print("-" * 50)
nn.fit(X_train_scaled, y_train_encoded, 
       epochs=200, 
       learning_rate=0.1, 
       loss='cross_entropy',
       verbose=False)

print(f"Training completed: 200 epochs")
print(f"Final training loss: {nn.loss_history[-1]:.6f}")

# Evaluate
y_pred_train = nn.predict(X_train_scaled)
y_pred_test = nn.predict(X_test_scaled)

# Convert predictions to class labels
y_pred_train_classes = np.argmax(y_pred_train, axis=1)
y_pred_test_classes = np.argmax(y_pred_test, axis=1)

train_accuracy = accuracy_score(y_train, y_pred_train_classes)
test_accuracy = accuracy_score(y_test, y_pred_test_classes)

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy:     {test_accuracy:.4f}")

# Show predictions for first 10 test samples
print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)
print(f"{'True Label':<12} {'Predicted':<12} {'Probabilities'}")
print("-" * 70)
for i in range(min(10, len(y_test))):
    probs = y_pred_test[i]
    print(f"{y_test[i]:<12} {y_pred_test_classes[i]:<12} {probs}")

# Loss history summary
print("\n" + "=" * 70)
print("TRAINING PROGRESS")
print("=" * 70)
print(f"Initial loss:  {nn.loss_history[0]:.6f}")
print(f"Loss at 50:    {nn.loss_history[49]:.6f}")
print(f"Loss at 100:   {nn.loss_history[99]:.6f}")
print(f"Loss at 150:   {nn.loss_history[149]:.6f}")
print(f"Final loss:    {nn.loss_history[-1]:.6f}")

print("\n" + "=" * 70)
print("Example completed successfully!")
print("=" * 70)
