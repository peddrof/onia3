import pandas as pd

# Read the actual answers from 'testanswers.csv'
actual = pd.read_csv('testanswers.csv')
print("Actual data columns:", actual.columns.tolist())
print("First few rows of actual data:\n", actual.head())

# Read the predictions from 'predictions.csv'
predicted = pd.read_csv('predictions.csv')
print("Predicted data columns:", predicted.columns.tolist())
print("First few rows of predicted data:\n", predicted.head())

# Calculate the number of correct predictions
# Assuming 'target' is the column name in predictions.csv
correct = (actual['target'] == predicted['target']).sum()

# Calculate the total number of predictions
total = len(actual)

# Calculate accuracy percentage
accuracy = (correct / total) * 100

# Create a DataFrame with the accuracy percentage
accuracy_df = pd.DataFrame({'accuracy_percentage': [accuracy]})

# Write the accuracy to 'accuracy.csv' without an index
accuracy_df.to_csv('accuracy.csv', index=False)
print(f"Accuracy: {accuracy:.2f}% saved to 'accuracy.csv'")