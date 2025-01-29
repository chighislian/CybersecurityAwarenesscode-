import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset from CSV
data = pd.read_csv(r"C:\Users\HP ProBook 440 G7\Desktop\myFinalProjec\Updated FUTO Cybersecurity Awareness Survey.csv")

# Handle missing values
missing_values = data.isnull().sum()
print("Missing values per column:\n", missing_values)

# Mapping each response to a numeric score
mapping_CybersecurityFamiliarity = {
    'Very familiar': 5,
    'Somewhat familiar': 3,
    'Not familiar': 0
}
data['CybersecurityFamiliarity'] = data['CybersecurityFamiliarity'].map(mapping_CybersecurityFamiliarity)

mapping_phishing = {
    'I have never heard of it': 0,
    "I have heard of it but don't know what it is": 5,
    'I know what it is and can i can identify phishing attempts': 10,
    'I know what it is but find it hard to identify phishing': 15
}
data['PhishingUnderstanding'] = data['PhishingUnderstanding'].map(mapping_phishing)

mapping_FirewallKnowledge = {
    'Yes, I know in detail': 10,
    'I have a basic understanding': 7,
    "I have heard of it but don't know how it works": 3,
    "No, I don't know": 0
}
data['FirewallKnowledge'] = data['FirewallKnowledge'].map(mapping_FirewallKnowledge)

mapping_PasswordUpdateFrequency = {
    'Monthly': 10,
    'Every few months': 7,
    'Once a year': 3,
    'Never': 0
}
data['PasswordUpdateFrequency'] = data['PasswordUpdateFrequency'].map(mapping_PasswordUpdateFrequency)

mapping_SamePasswordUsage = {
    'Yes, for most accounts': 10,
    'Yes, but only for a few accounts': 5,
    'No, I use different passwords for all accounts': 0
}
data['SamePasswordUsage'] = data['SamePasswordUsage'].map(mapping_SamePasswordUsage)

mapping_TwoFactorAuthUsage = {
    'Yes, for all accounts': 10,
    'Yes, but only for some accounts': 5,
    'No': 0
}
data['TwoFactorAuthUsage'] = data['TwoFactorAuthUsage'].map(mapping_TwoFactorAuthUsage)

mapping_UnknownEmailResponse = {
    'Open the attachment to see what it is': 0,
    'Delete the email immediately': 5,
    'Scan the attachment with antivirus software before opening': 7,
    'Report it as spam/phishing': 10
}
data['UnknownEmailResponse'] = data['UnknownEmailResponse'].map(mapping_UnknownEmailResponse)

mapping_CyberIncidentExperience = {
    'Yes': 0,
    'No': 1
}
data['CyberIncidentExperience'] = data['CyberIncidentExperience'].map(mapping_CyberIncidentExperience)

mapping_IncidentResponse = {
    'Changed passwords': 1,
    'Reported it to the relevant authorities': 2,
    'Did nothing': 0,
    'Contacted the service provider': 3
}
data['IncidentResponse'] = data['IncidentResponse'].map(mapping_IncidentResponse)

mapping_PhishingEmailResponse = {
    'Click the link and update your information': 0,
    'Ignore the email': 5,
    'Mark the email as spam': 7,
    'Contact the bank directly to verify the email': 10
}
data['PhishingEmailResponse'] = data['PhishingEmailResponse'].map(mapping_PhishingEmailResponse)

mapping_PopUpAlertAction = {
    'Run your antivirus software to check for issues': 10,
    'Close the pop-up and continue browsing': 7,
    'Restart your computer': 5,
    'Download and install the software immediately': 0
}
data['PopUpAlertAction'] = data['PopUpAlertAction'].map(mapping_PopUpAlertAction)

mapping_CybersecurityImportance = {
    'Extremely important': 5,
    'Very important': 3,
    'Somewhat important': 1,
    'Not important': 0
}
data['CybersecurityImportance'] = data['CybersecurityImportance'].map(mapping_CybersecurityImportance)

mapping_DataProtectionConfidence = {
    'Very confident': 5,
    'Somewhat confident': 3,
    'Not very confident': 1,
    'Not confident at all': 0
}
data['DataProtectionConfidence'] = data['DataProtectionConfidence'].map(mapping_DataProtectionConfidence)

mapping_CyberTrainingInterest = {
    'Yes': 5,
    'Maybe': 3,
    'No': 0
}
data['CyberTrainingInterest'] = data['CyberTrainingInterest'].map(mapping_CyberTrainingInterest)

# Create TotalScore by summing up the mapped columns
data['TotalScore'] = data[['CybersecurityFamiliarity', 'PhishingUnderstanding', 'FirewallKnowledge',
                           'SamePasswordUsage', 'TwoFactorAuthUsage', 'UnknownEmailResponse',
                           'CyberIncidentExperience', 'IncidentResponse', 'PhishingEmailResponse',
                           'PopUpAlertAction', 'CybersecurityImportance', 'DataProtectionConfidence',
                           'CyberTrainingInterest', 'PasswordUpdateFrequency']].sum(axis=1)

# Maximum possible score
max_score = 109
data['CyberAwarenessPercentage'] = (data['TotalScore'] / max_score) * 100

# Create AwarenessLevel based on CyberAwarenessPercentage
bins = [0, 40, 70, 100]
labels = ['Low Awareness', 'Medium Awareness', 'High Awareness']
data['AwarenessLevel'] = pd.cut(data['CyberAwarenessPercentage'], bins=bins, labels=labels)

# Feature variables (X) and target variable (Y)
X = data.drop(['TotalScore', 'CyberAwarenessPercentage', 'AwarenessLevel'], axis=1)
Y = data['AwarenessLevel']

# Imputation of missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Encode the target variable (AwarenessLevel) with LabelEncoder
le = LabelEncoder()
Y_encoded = le.fit_transform(Y)

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(X_imputed, Y_encoded, test_size=0.2, random_state=42)

# Naive Bayes classifier (GaussianNB)
nb_model = GaussianNB()

# Train the Naive Bayes model
nb_model.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = nb_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("Classification Report:")
print(classification_report(Y_test, Y_pred))

print("Confusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))

# Save the trained model to a file
joblib.dump(nb_model, 'cyber_awareness_nb_model.pkl')

# Plotting the confusion matrix (Optional)
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(Y_test, Y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Naive Bayes')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()