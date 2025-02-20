



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# Load your dataset (replace 'your_dataset.csv' with your actual file)
# Assume the dataset has features X and target y with cybersecurity awareness levels
data = pd.read_csv(r"C:\Users\HP ProBook 440 G7\Desktop\myFinalProjec\Updated FUTO Cybersecurity Awareness Survey.csv")

# Separate features and target
X = data.drop(columns=["TotalScore",'CyberAwarenessPercentage','AwarenessLevel'])  # Replace with your target column name
Y = data["AwarenessLevel"]

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, Y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Calculate precision, recall, and F1-score by class
metrics = precision_recall_fscore_support(Y_test, y_pred, average=None, labels=rf_model.classes_)

# Extract metrics
precision = metrics[0]
recall = metrics[1]
f1_score = metrics[2]
classes = rf_model.classes_

# Plot the metrics
x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, precision, width, label='Precision', color='skyblue')
ax.bar(x, recall, width, label='Recall', color='lightgreen')
ax.bar(x + width, f1_score, width, label='F1-Score', color='coral')

# Add labels, title, and legend
ax.set_xlabel('Cybersecurity Awareness Levels')
ax.set_ylabel('Score')
ax.set_title('Random Forest Metrics by Awareness Level')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()