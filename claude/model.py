import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Step 1: Load the training data
print("Step 1: Loading training data...")
train_data = pd.read_csv('train.csv')
print(f"Loaded {len(train_data)} rows of training data")
print(train_data.head())

# Step 2: Separate features and target
print("\nStep 2: Separating features and target...")
X = train_data.drop(['id', 'target'], axis=1)  # Features (all columns except id and target)
y = train_data['target']  # Target variable
print(f"Feature shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Step 3: Split data into training and validation sets
print("\nStep 3: Splitting data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")

# Step 4: Scale the features (important for neural networks)
print("\nStep 4: Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
print("Feature scaling complete")

# Step 5: Create and train the neural network
print("\nStep 5: Training the neural network...")
# Create a Multi-Layer Perceptron (neural network)
# hidden_layer_sizes: two hidden layers with 50 and 25 neurons
# max_iter: maximum number of iterations
# random_state: for reproducibility
mlp = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)

# Train the model
mlp.fit(X_train_scaled, y_train)
print("Neural network training complete")

# Step 6: Evaluate the model
print("\nStep 6: Evaluating the model...")
train_score = mlp.score(X_train_scaled, y_train)
val_score = mlp.score(X_val_scaled, y_val)
print(f"Training accuracy: {train_score:.4f}")
print(f"Validation accuracy: {val_score:.4f}")

# Make predictions on validation set
y_pred = mlp.predict(X_val_scaled)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Create confusion matrix
conf_matrix = confusion_matrix(y_val, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Step 7: Visualize confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
classes = np.unique(y)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add text annotations
thresh = conf_matrix.max() / 2
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix visualization saved as 'confusion_matrix.png'")

# Step 8: Load the test data
print("\nStep 8: Loading test data...")
test_data = pd.read_csv('test.csv')
print(f"Loaded {len(test_data)} rows of test data")
print(test_data.head())

# Step 9: Prepare test data
print("\nStep 9: Preparing test data...")
test_ids = test_data['id']
X_test = test_data.drop(['id'], axis=1)  # Features
X_test_scaled = scaler.transform(X_test)  # Scale using the same scaler
print("Test data preparation complete")

# Step 10: Make predictions on test data
print("\nStep 10: Making predictions on test data...")
test_predictions = mlp.predict(X_test_scaled)
print(f"Generated {len(test_predictions)} predictions")

# Step 11: Create submission file
print("\nStep 11: Creating submission file...")
submission = pd.DataFrame({
    'id': test_ids,
    'target': test_predictions
})
submission.to_csv('predictions.csv', index=False)
print("Predictions saved to 'predictions.csv'")

print("\nProcess complete!")