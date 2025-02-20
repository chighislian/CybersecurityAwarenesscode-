import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv(r"C:\Users\HP ProBook 440 G7\Desktop\myFinalProjec\Updated FUTO Cybersecurity Awareness Survey.csv")

# Check for missing values and data types
print(data.isnull().sum())
print(data.dtypes)

# Example preprocessing: Mapping categorical variables to numerical ones (ensure it's done beforehand)
# Ensure your columns are all numeric before fitting models

# Mapping each response to numeric score to get the awareness total
mapping_CybersecurityFamiliarity = {'Very familiar': 5, 'Somewhat familiar': 3, 'Not familiar': 0}
data['CybersecurityFamiliarity'] = data['CybersecurityFamiliarity'].map(mapping_CybersecurityFamiliarity)

mapping_phishing = {'I have never heard of it': 0, "I have heard of it but don't know what it is": 5,
                    'I know what it is and can i can identify phishing attempts': 10,
                    'I know what it is but find it hard to identify phishing': 15}
data['PhishingUnderstanding'] = data['PhishingUnderstanding'].map(mapping_phishing)

mapping_FirewallKnowledge = {'Yes, I know in detail': 10, 'I have a basic understanding': 7,
                            "I have heard of it but don't know how it works": 3, "No, I don't know": 0}
data['FirewallKnowledge'] = data['FirewallKnowledge'].map(mapping_FirewallKnowledge)

mapping_PasswordUpdateFrequency = {'Monthly': 10, 'Every few months': 7, 'Once a year': 3, 'Never': 0}
data['PasswordUpdateFrequency'] = data['PasswordUpdateFrequency'].map(mapping_PasswordUpdateFrequency)

mapping_SamePasswordUsage = {'Yes, for most accounts': 10, 'Yes, but only for a few accounts': 5,
                             'No, I use different passwords for all accounts': 0}
data['SamePasswordUsage'] = data['SamePasswordUsage'].map(mapping_SamePasswordUsage)

mapping_TwoFactorAuthUsage = {'Yes, for all accounts': 10, 'Yes, but only for some accounts': 5, 'No': 0}
data['TwoFactorAuthUsage'] = data['TwoFactorAuthUsage'].map(mapping_TwoFactorAuthUsage)

mapping_UnknownEmailResponse = {'Open the attachment to see what it is': 0, 'Delete the email immediately': 5,
                                'Scan the attachment with antivirus software before opening': 7,
                                'Report it as spam/phishing': 10}
data['UnknownEmailResponse'] = data['UnknownEmailResponse'].map(mapping_UnknownEmailResponse)

mapping_CyberIncidentExperience = {'Yes': 0, 'No': 1}
data['CyberIncidentExperience'] = data['CyberIncidentExperience'].map(mapping_CyberIncidentExperience)

mapping_IncidentResponse = {'Changed passwords': 1, 'Reported it to the relevant authorities': 2, 'Did nothing': 0,
                            'Contacted the service provider': 3}
data['IncidentResponse'] = data['IncidentResponse'].map(mapping_IncidentResponse)

mapping_PhishingEmailResponse = {'Clink the link and update your information': 0, 'Ignore the email': 5,
                                 'Mark the email as spam': 7, 'Contact the bank directly to verify the email': 10}
data['PhishingEmailResponse'] = data['PhishingEmailResponse'].map(mapping_PhishingEmailResponse)

mapping_PopUpAlertAction = {'Run your antivirus software to check for issues': 10, 'Close the pop-up and continue browsing': 7,
                            'Restart your computer': 5, 'Download and install the software immediately': 0}
data['PopUpAlertAction'] = data['PopUpAlertAction'].map(mapping_PopUpAlertAction)

mapping_CybersecurityImportance = {'Extremely important': 5, 'Very important': 3, 'Somewhat important': 1, 'Not important': 0}
data['CybersecurityImportance'] = data['CybersecurityImportance'].map(mapping_CybersecurityImportance)

mapping_DataProtectionConfidence = {'Very confident': 5, 'Somewhat confident': 3, 'Not very confident': 1,
                                    'Not confident at all': 0}
data['DataProtectionConfidence'] = data['DataProtectionConfidence'].map(mapping_DataProtectionConfidence)

mapping_CyberTrainingInterest = {'Yes': 5, 'Maybe': 3, 'No': 0}
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

# Creating the target variable based on the awareness percentage
bins = [0, 40, 70, 100]
labels = ['Low Awareness', 'Medium Awareness', 'High Awareness']
data['AwarenessLevel'] = pd.cut(data['CyberAwarenessPercentage'], bins=bins, labels=labels)

# Label encoding for categorical target variable
label_encoder = LabelEncoder()
data['AwarenessLevel'] = label_encoder.fit_transform(data['AwarenessLevel'])

# Check for missing values and impute if necessary
print(data.isnull().sum())

# Separate features (X) and target variable (Y)
X = data.drop(['AwarenessLevel', 'TotalScore', 'CyberAwarenessPercentage'], axis=1)
Y = data['AwarenessLevel']

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize and fit the Logistic Regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, Y_train)

# Make predictions
Y_pred = logreg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(Y_test, Y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

# Calculate precision, recall, and F1-score for each awareness level
precision, recall, f1_score, _ = precision_recall_fscore_support(Y_test, Y_pred, labels=logreg.classes_)

# Awareness level labels
awareness_levels = ['Low Awareness', 'Medium Awareness', 'High Awareness']

# Create a bar plot
x = np.arange(len(awareness_levels))
width = 0.25

plt.figure(figsize=(10, 6))

# Plot precision, recall, and F1-score
plt.bar(x - width, precision, width, label='Precision', color='skyblue')
plt.bar(x, recall, width, label='Recall', color='lightgreen')
plt.bar(x + width, f1_score, width, label='F1-Score', color='coral')

# Add labels, title, and legend
plt.xlabel('Cybersecurity Awareness Levels')
plt.ylabel('Score')
plt.title('Logistic Regression Metrics by Cybersecurity Awareness Level')
plt.xticks(x, awareness_levels)
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()


#confusion matrix

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix
conf_matrix = confusion_matrix(Y_test, Y_pred)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)

# Add labels, title, and axis ticks
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Logistic Regression ')

# Display the plot
plt.show()