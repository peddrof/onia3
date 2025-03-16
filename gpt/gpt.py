import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# ----------------------------
# 1. Load the training data
# ----------------------------
train_data = pd.read_csv('train.csv')
print("First few rows of training data:")
print(train_data.head())

# ----------------------------
# 2. Separate features and target
# ----------------------------
# Feature columns are "col_0" to "col_12". The "id" column is not used for training.
feature_columns = [f"col_{i}" for i in range(13)]
X_train = train_data[feature_columns]
y_train = train_data["target"]

print("Training features shape:", X_train.shape)
print("Training target shape:", y_train.shape)

# ----------------------------
# 3. Scale the features
# ----------------------------
# Standardizing features is recommended for neural networks.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ----------------------------
# 4. Build and train the neural network model
# ----------------------------
# MLPClassifier is used here with one hidden layer of 100 neurons.
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                    max_iter=300, random_state=42)
mlp.fit(X_train_scaled, y_train)

# ----------------------------
# 5. Load and prepare the test data
# ----------------------------
# The test CSV file is assumed to have an "id" column and the 13 feature columns.
test_data = pd.read_csv('test.csv')
X_test = test_data[feature_columns]
ids = test_data['id']
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 6. Make predictions on the test data
# ----------------------------
predictions = mlp.predict(X_test_scaled)
print("First 10 predictions:", predictions[:10])

# ----------------------------
# 7. Save the predictions to a CSV file
# ----------------------------
# The output file "predictions.csv" will contain two columns: "id" and "target".
output_df = pd.DataFrame({
    "id": ids,
    "target": predictions
})
output_df.to_csv("predictions.csv", index=False)

print("Predictions saved to predictions.csv")
