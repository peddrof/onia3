# Import the necessary libraries
import pandas as pd  # For reading and managing CSV files
from sklearn.neural_network import MLPClassifier  # The neural network model
from sklearn.preprocessing import StandardScaler  # To scale the features
import numpy as np  # For checking unique values (optional)

# Step 1: Load the training data from 'train.csv'
train_data = pd.read_csv('train.csv')
print("Training data shape:", train_data.shape)  # Should be (10500, 15)

# Step 2: Define the feature columns (col_0 to col_12)
feature_columns = ['col_' + str(i) for i in range(13)]  # Creates ['col_0', 'col_1', ..., 'col_12']

# Step 3: Extract features and target from training data
X_train = train_data[feature_columns]  # Features: the 13 numeric columns
y_train = train_data['target']  # Target: the column we want to predict
print("Features shape:", X_train.shape)  # Should be (10500, 13)
print("Target shape:", y_train.shape)  # Should be (10500,)

# Step 4: Check for missing values (just to be safe)
print("Missing values in training data:", train_data.isnull().sum().sum())
# If this prints anything other than 0, we’d need to handle missing data, but let’s assume it’s clean

# Step 5: Scale the features (important for neural networks)
scaler = StandardScaler()  # Creates a scaler to standardize the data
scaler.fit(X_train)  # Learns the mean and standard deviation from training features
X_train_scaled = scaler.transform(X_train)  # Scales the training features

# Step 6: Create and train the neural network
model = MLPClassifier(
    hidden_layer_sizes=(100,),  # One hidden layer with 100 neurons
    max_iter=1000,  # Maximum number of iterations to train
    random_state=42  # Ensures consistent results
)
model.fit(X_train_scaled, y_train)  # Trains the model on scaled features and target
print("Model training complete!")

# Step 7: Load the test data from 'test.csv'
test_data = pd.read_csv('test.csv')
print("Test data shape:", test_data.shape)  # Should be (some_number, 14)

# Step 8: Extract features from test data
X_test = test_data[feature_columns]  # Same 13 feature columns
print("Test features shape:", X_test.shape)

# Step 9: Scale the test features using the same scaler
X_test_scaled = scaler.transform(X_test)  # Use the training scaler, don’t refit!

# Step 10: Predict the target values for the test data
y_pred = model.predict(X_test_scaled)  # Generates predictions (0 to 4)
print("Unique predicted values:", np.unique(y_pred))  # Should be some of [0, 1, 2, 3, 4]

# Step 11: Create a results DataFrame with 'id' and 'target'
results = pd.DataFrame({
    'id': test_data['id'],  # IDs from the test file
    'target': y_pred  # Predicted targets
})
print("First few predictions:\n", results.head())

# Step 12: Save the results to 'predictions.csv'
results.to_csv('predictions.csv', index=False)  # No extra index column
print("Predictions saved to 'predictions.csv'!")