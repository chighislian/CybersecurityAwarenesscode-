import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
data = pd.read_csv(r"C:\Users\HP ProBook 440 G7\Desktop\myFinalProjec\Updated FUTO Cybersecurity Awareness Survey.csv")

# Feature selection and target variable
# Replace 'target_column' with the actual column name for the labels

X = data.drop(['TotalScore', 'CyberAwarenessPercentage', 'AwarenessLevel'], axis=1)
Y = data['AwarenessLevel']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the trained model for later use
joblib.dump(rf_model, 'random_forest_model.pkl')
print("Model saved as 'random_forest_model.pkl'")























import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
data = pd.read_csv(r"C:\Users\HP ProBook 440 G7\Desktop\myFinalProjec\Updated FUTO Cybersecurity Awareness Survey.csv")
# Feature selection and target variable
X = data.drop(['TotalScore', 'CyberAwarenessPercentage', 'AwarenessLevel'], axis=1)
Y = data['AwarenessLevel']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

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
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Hyperparameters:", best_params)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the best model for later use
joblib.dump(best_model, 'decision_tree_best_model.pkl')
print("Best model saved as 'decision_tree_best_model.pkl'")










# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
data = pd.read_csv(r"C:\Users\HP ProBook 440 G7\Desktop\myFinalProjec\Updated FUTO Cybersecurity Awareness Survey.csv")

# Feature selection and target variable
X = data.drop(['TotalScore', 'CyberAwarenessPercentage', 'AwarenessLevel'], axis=1)
Y = data['AwarenessLevel']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=y)

# Define the SVM classifier
svm_model = SVC(random_state=42)

# Define the hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],                # Regularization parameter
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type
    'gamma': ['scale', 'auto', 0.01, 0.001],         # Kernel coefficient
    'degree': [2, 3, 4],                   # Degree for the 'poly' kernel
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid,
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
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the best model for later use
joblib.dump(best_model, 'svm_best_model.pkl')
print("Best model saved as 'svm_best_model.pkl'")









# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load your dataset
# Replace 'your_dataset.csv' with the actual path to your dataset
data = pd.read_csv('your_dataset.csv')

# Feature selection and target variable
# Replace 'target_column' with the actual column name for the labels
X = data.drop(columns=['target_column'])
y = data['target_column']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Define the Logistic Regression model
lr_model = LogisticRegression(random_state=42, max_iter=1000)

# Define the hyperparameter grid
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],  # Regularization type
    'C': [0.01, 0.1, 1, 10, 100],                  # Inverse of regularization strength
    'solver': ['lbfgs', 'liblinear', 'saga'],      # Algorithm to use for optimization
    'l1_ratio': [0, 0.5, 1]                        # Only relevant for 'elasticnet' penalty
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=lr_model, param_grid=param_grid,
                           scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

# Fit the model to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Hyperparameters:", best_params)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the best model for later use
joblib.dump(best_model, 'logistic_regression_best_model.pkl')
print("Best model saved as 'logistic_regression_best_model.pkl'")











# Define the Logistic Regression model
lr_model = LogisticRegression(random_state=42, max_iter=1000)

# Define the hyperparameter grid
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],  # Regularization type
    'C': [0.01, 0.1, 1, 10, 100],                  # Inverse of regularization strength
    'solver': ['lbfgs', 'liblinear', 'saga'],      # Algorithm to use for optimization
    'l1_ratio': [0, 0.5, 1]                        # Only relevant for 'elasticnet' penalty
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=lr_model, param_grid=param_grid,
                           scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

# Fit the model to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Hyperparameters:", best_params)












# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

data = pd.read_csv('cybersecurity_data.csv')

# Feature selection and target variable

X = data.drop(columns=['awareness_level'])  # Features
y = data['awareness_level']                # Target

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

# Scale the features (important for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the KNN model
knn_model = KNeighborsClassifier()

# Define the hyperparameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],          # Number of neighbors
    'weights': ['uniform', 'distance'],       # Weight function
    'metric': ['euclidean', 'manhattan'],     # Distance metric
    'p': [1, 2]                               # Power parameter for Minkowski
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=knn_model, param_grid=param_grid,
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
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(Y_test, y_pred))

# Save the best model and scaler for future use
joblib.dump(best_model, 'knn_cybersecurity_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Best model saved as 'knn_cybersecurity_model.pkl' and scaler saved as 'scaler.pkl'")

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
print("\nAccuracy Score:", accuracy_score(Y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the best model for later use
joblib.dump(best_model, 'logistic_regression_best_model.pkl')
print("Best model saved as 'logistic_regression_best_model.pkl'")