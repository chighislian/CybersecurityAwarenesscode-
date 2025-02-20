import numpy as np
import pandas as pd
import joblib


from sklearn.tree import DecisionTreeClassifier



import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# load the dataset from excel using pandas
data = pd.read_csv(r"C:\Users\HP ProBook 440 G7\Desktop\myFinalProjec\Updated FUTO Cybersecurity Awareness Survey.csv")
# print(data.head())

# check for missing values / cleaning the dataSet
missing_values = data.isnull()  # This method returns  True where data is missing and False where data is present.
print(missing_values)
missing_values_per_column = data.isnull().sum()  # This returns the number of missing values per column
print(missing_values_per_column)
print(data.dtypes)  # Use the dtypes attribute to check the data types of each column.
data.info()  # Use this method to get a more detailed summary including data types and non-null counts for each column
# No missing value found

# mapping each response to numeric score To get the awareness Total

# Mapping for CybersecurityFamiliarity
mapping_CybersecurityFamiliarity = {
    'Very familiar': 5,
    'Somewhat familiar': 3,
    'Not familiar': 0
}
data['CybersecurityFamiliarity'] = data['CybersecurityFamiliarity'].map(mapping_CybersecurityFamiliarity)

# Mapping for PhishingUnderstanding
mapping_phishing = {
    'I have never heard of it': 0,
    "I have heard of it but don't know what it is": 5,
    'I know what it is and I can identify phishing attempts': 15,
    'I know what it is but find it hard to identify phishing': 10
}
data['PhishingUnderstanding'] = data['PhishingUnderstanding'].map(mapping_phishing)

# Mapping for FirewallKnowledge
mapping_FirewallKnowledge = {
    'Yes, I know in detail': 10,
    'I have a basic understanding': 7,
    "I have heard of it but don't know how it works": 3,
    "No, I don't know": 0
}
data['FirewallKnowledge'] = data['FirewallKnowledge'].map(mapping_FirewallKnowledge)

# Mapping for PasswordUpdateFrequency
mapping_PasswordUpdateFrequency = {
    'Monthly': 10,
    'Every few months': 7,
    'Once a year': 3,
    'Never': 0
}
data['PasswordUpdateFrequency'] = data['PasswordUpdateFrequency'].map(mapping_PasswordUpdateFrequency)

# Mapping for SamePasswordUsage
mapping_SamePasswordUsage = {
    'Yes, for most accounts': 0,
    'Yes, but only for a few accounts': 5,
    'No, I use different passwords for all accounts': 10
}
data['SamePasswordUsage'] = data['SamePasswordUsage'].map(mapping_SamePasswordUsage)

# Mapping for TwoFactorAuthUsage
mapping_TwoFactorAuthUsage = {
    'Yes, for all accounts': 10,
    'Yes, but only for some accounts': 5,
    'No': 0
}
data['TwoFactorAuthUsage'] = data['TwoFactorAuthUsage'].map(mapping_TwoFactorAuthUsage)

# Mapping for UnknownEmailResponse
mapping_UnknownEmailResponse = {
    'Open the attachment to see what it is': 0,
    'Delete the email immediately': 5,
    'Scan the attachment with antivirus software before opening': 7,
    'Report it as spam/phishing': 10
}
data['UnknownEmailResponse'] = data['UnknownEmailResponse'].map(mapping_UnknownEmailResponse)

# Mapping for CyberIncidentExperience
mapping_CyberIncidentExperience = {
    'Yes': 1,  # Changed from 0 to 1 (since experience matters)
    'No': 0
}
data['CyberIncidentExperience'] = data['CyberIncidentExperience'].map(mapping_CyberIncidentExperience)

# Mapping for IncidentResponse
mapping_IncidentResponse = {
    'Changed passwords': 1,
    'Reported it to the relevant authorities': 3,
    'Did nothing': 0,
    'Contacted the service provider': 2
}
data['IncidentResponse'] = data['IncidentResponse'].map(mapping_IncidentResponse)

# Mapping for PhishingEmailResponse
mapping_PhishingEmailResponse = {
    'Click the link and update your information': 0,  # Fixed typo
    'Ignore the email': 5,
    'Mark the email as spam': 7,
    'Contact the bank directly to verify the email': 10
}
data['PhishingEmailResponse'] = data['PhishingEmailResponse'].map(mapping_PhishingEmailResponse)

# Mapping for PopUpAlertAction
mapping_PopUpAlertAction = {
    'Run your antivirus software to check for issues': 10,
    'Close the pop-up and continue browsing': 7,
    'Restart your computer': 5,
    'Download and install the software immediately': 0
}
data['PopUpAlertAction'] = data['PopUpAlertAction'].map(mapping_PopUpAlertAction)

# Mapping for CybersecurityImportance
mapping_CybersecurityImportance = {
    'Extremely important': 5,
    'Very important': 3,
    'Somewhat important': 1,
    'Not important': 0
}
data['CybersecurityImportance'] = data['CybersecurityImportance'].map(mapping_CybersecurityImportance)

# Mapping for DataProtectionConfidence
mapping_DataProtectionConfidence = {
    'Very confident': 5,
    'Somewhat confident': 3,
    'Not very confident': 1,
    'Not confident at all': 0
}
data['DataProtectionConfidence'] = data['DataProtectionConfidence'].map(mapping_DataProtectionConfidence)

# Mapping for CyberTrainingInterest
mapping_CyberTrainingInterest = {
    'Yes': 5,
    'Maybe': 3,
    'No': 0
}
data['CyberTrainingInterest'] = data['CyberTrainingInterest'].map(mapping_CyberTrainingInterest)



# Trying to Sum up the mapping score for each student

data['TotalScore'] = data[['CybersecurityFamiliarity', 'PhishingUnderstanding', 'FirewallKnowledge',
                           'SamePasswordUsage', 'TwoFactorAuthUsage', 'UnknownEmailResponse',
                           'CyberIncidentExperience', 'IncidentResponse', 'PhishingEmailResponse',
                           'PopUpAlertAction', 'CybersecurityImportance', 'DataProtectionConfidence',
                           'CyberTrainingInterest', 'PasswordUpdateFrequency']].sum(axis=1)

# Maximum possible score
max_score = 109

# Calculate the CyberAwarenessPercentage
data['CyberAwarenessPercentage'] = (data['TotalScore'] / max_score) * 100

# Print TotalScore and CyberAwarenessPercentage
print(data[['TotalScore', 'CyberAwarenessPercentage']])

# Creating the target variable based on the awareness percentage
bins = [0, 40, 70, 100]
labels = ['Low Awareness',  'Medium Awareness', 'High Awareness']
data['AwarenessLevel'] = pd.cut(data['CyberAwarenessPercentage'], bins=bins, labels=labels)
print(data['AwarenessLevel'])

# this step we label  the data into feature X and Target Y
X = data.drop(['TotalScore', 'CyberAwarenessPercentage', 'AwarenessLevel'], axis=1)
Y = data['AwarenessLevel']

# This step we split the data into both the Training (80%) and Testing (20%)
X_train,  X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# This step we choose the model that is suitable for project based on the dataset we're working on
# We choose Random Forest Classifier model





# Define the Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)

# Define the hyperparameter grid
param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],  # Criterion to measure quality of split
    'max_depth': [None, 10, 20, 30],              # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],              # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],                # Minimum samples required to be a leaf node
    'max_features': [None, 'sqrt', 'log2'],       # Number of features to consider for the best split
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid,
                           scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

# Fit the model to the training data
grid_search.fit(X_train, Y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Hyperparameters:", best_params)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
print("\nAccuracy Score:", accuracy_score(Y_test, y_pred))
print("\nClassification Report:\n", classification_report(Y_test, y_pred))

# Save the best model for later use
joblib.dump(best_model, 'decision_tree_best_model.pkl')
print("Best model saved as 'decision_tree_best_model.pkl'")













# Define the Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)

# Define the hyperparameter grid
param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],  # Criterion to measure quality of split
    'max_depth': [None, 10, 20, 30],              # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],              # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],                # Minimum samples required to be a leaf node
    'max_features': [None, 'sqrt', 'log2'],       # Number of features to consider for the best split
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid,
                           scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

# Fit the model to the training data
grid_search.fit(X_train, Y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Hyperparameters:", best_params)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
print("\nAccuracy Score:", accuracy_score(Y_test, y_pred))
print("\nClassification Report:\n", classification_report(Y_test, y_pred))

# Save the best model for later use
joblib.dump(best_model, 'decision_tree_best_model.pkl')
print("Best model saved as 'decision_tree_best_model.pkl'")








import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

# Assuming your Decision Tree model is trained and you have predictions
# Y_test is the true labels, and y_pred is the predicted labels from the Decision Tree model
# best_model.classes_ gives the class labels

# Calculate precision, recall, and F1-score for each class
precision, recall, f1_score, _ = precision_recall_fscore_support(Y_test, y_pred, labels=best_model.classes_)

# Class labels
classes = best_model.classes_

# Create a bar plot
x = np.arange(len(classes))
width = 0.25

plt.figure(figsize=(10, 6))

# Plot precision, recall, and F1-score
plt.bar(x - width, precision, width, label='Precision', color='skyblue')
plt.bar(x, recall, width, label='Recall', color='lightgreen')
plt.bar(x + width, f1_score, width, label='F1-Score', color='coral')

# Add labels, title, and legend
plt.xlabel('Cybersecurity Awareness Levels')
plt.ylabel('Score')
plt.title('Decision Tree Metrics by Class')
plt.xticks(x, classes)
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()






                                    #CONFUSION MATRIX FOR DECISION TREE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Predict using the trained Decision Tree model
y_pred_dt = best_model.predict(X_test)  # Use the best decision tree model after GridSearchCV

# Compute the confusion matrix
cm = confusion_matrix(Y_test, y_pred_dt)

# Define class labels (from your awareness levels)
class_labels = ['Low Awareness', 'Medium Awareness', 'High Awareness']

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=class_labels, yticklabels=class_labels)

# Add labels, title, and axis names
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Decision Tree Classifier')
plt.show()




from collections import Counter

# Count occurrences of each class
actual_counts = Counter(Y_test)
predicted_counts = Counter(y_pred)

# Get unique class labels
labels = list(set(Y_test) | set(y_pred))

# Get counts in same order
actual_values = [actual_counts[label] for label in labels]
predicted_values = [predicted_counts[label] for label in labels]

# Set bar width
bar_width = 0.35
x = np.arange(len(labels))

# Create bar chart
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - bar_width/2, actual_values, bar_width, label="Actual", color='blue')
ax.bar(x + bar_width/2, predicted_values, bar_width, label="Predicted", color='orange')

# Labels and title
ax.set_xlabel("Cybersecurity Awareness Levels")
ax.set_ylabel("Number of Students")
ax.set_title("Comparison of Actual vs. Predicted Results")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Show the chart
plt.show()