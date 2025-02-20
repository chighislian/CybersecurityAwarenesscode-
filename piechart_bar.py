
import pandas as pd
import matplotlib.pyplot as plt
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


import pandas as pd



# Count occurrences of each awareness level
awareness_counts = data['AwarenessLevel'].value_counts()

# Convert counts into a format for plotting
categories = awareness_counts.index.tolist()  # Awareness levels
counts = awareness_counts.values.tolist()     # Corresponding counts

print(categories)  # Example output: ['Medium', 'Low', 'High']
print(counts)      # Example output: [120, 50, 30]


# Sample data
categories = ['Low', 'Medium', 'High']
counts = [98, 406, 46]  # Example numbers

# --- Pie Chart ---
plt.figure(figsize=(6,6))
plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=140, colors=['red', 'orange', 'green'])
#plt.title('Cybersecurity Awareness Distribution (Pie Chart)')
plt.savefig('awareness_pie_chart.png')  # Save as an image file
plt.show()

# --- Bar Chart ---
plt.figure(figsize=(6,4))
plt.bar(categories, counts, color=['red', 'orange', 'green'])
#plt.title('Cybersecurity Awareness Distribution (Bar Chart)')
plt.xlabel('Awareness Level')
plt.ylabel('Number of Students')
plt.savefig('awareness_bar_chart.png')  # Save as an image file
plt.show()





#SPEEDOMETER
import matplotlib.pyplot as plt
import numpy as np

# Define the gauge sections
fig, ax = plt.subplots(figsize=(6, 3))
ax.set_xlim(0, 100)  # Percentage scale (0% to 100%)
ax.set_ylim(0, 1)

# Define color zones
ax.barh(0.5, 100, height=0.3, color="lightgray")  # Background
ax.barh(0.5, 50, height=0.3, color="red")  # Low zone (0-50%)
ax.barh(0.5, 30, left=50, height=0.3, color="yellow")  # Medium (50-80%)
ax.barh(0.5, 20, left=80, height=0.3, color="green")  # High (80-100%)

# Add indicator (89.19%)
arrow_x = 89.19
ax.scatter(arrow_x, 0.5, color="black", s=200, label=f"Accuracy: {arrow_x:.2f}%")

# Remove axes
ax.set_xticks([])
ax.set_yticks([])
ax.spines[:].set_visible(False)

# Add title
plt.title("Model Accuracy Gauge Chart", fontsize=14)
plt.legend()
plt.show()