# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 12:01:54 2024

@author: avina
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_hc = pd.read_csv('P:/My Documents/Books & Research/Analytics Vidya Blackbelt program/Building your 1st ML Model/aJ17yZkR3SQbeSq7Aobq_Healthcare_Data_Preprocessed.csv')
df_hc.columns
df_hc.dtypes
df_hc.shape
df_hc.head()

missing_values = df_hc.isnull().sum()
print(missing_values)

for col in ['Diet_Type_Vegan', 'Diet_Type_Vegetarian', 'Blood_Group_AB', 'Blood_Group_B','Blood_Group_O']: 
    if df_hc[col].dtype == 'bool':
        df_hc[col] = df_hc[col].astype(int)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

X = df_hc.drop(columns=['Target'])
y = df_hc['Target']

# Split the data into training and testing sets (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Normalize the data for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train a K-Nearest Neighbors classifier
k = 5 
knn_model = KNeighborsClassifier(n_neighbors=5)



# K-Fold Cross-Validation
from sklearn.model_selection import train_test_split, KFold, cross_val_score
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(knn_model, X_train_scaled, y_train, cv=kf, scoring='accuracy')

# Print cross-validation results
print("Cross-Validation Scores:", cv_scores)
print(f"Mean Accuracy from Cross-Validation: {np.mean(cv_scores) * 100:.2f}%\n")


knn_model.fit(X_train_scaled, y_train)
y_pred = knn_model.predict(X_test_scaled)


# Evaluate the model's performance
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

# Classification report
report = classification_report(y_test, y_pred, target_names=['Healthy', 'Unhealthy'])
print("Classification Report:\n", report)

# Confusion matrix for visual evaluation
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Unhealthy'], yticklabels=['Healthy', 'Unhealthy'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plotting accuracy vs. K value
k_range = range(1, 21)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=kf, scoring='accuracy')
    k_scores.append(np.mean(scores))

# Plot k-values vs Cross-Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(k_range, k_scores, marker='o')
plt.title('K-Value vs Cross-Validation Accuracy')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validation Accuracy')
plt.grid()
plt.show()

error_rate = []
specificity_values = []


k_values = range(1, 21)  
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    y_pred = knn.predict(X_test_scaled)
    
    # Calculate error rate
    error_rate.append(np.mean(y_pred != y_test))
        
    # Calculate specificity (True Negative Rate)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificity_values.append(specificity)


plt.figure(figsize=(14, 6))

# Error Rate Plot
plt.subplot(1, 2, 1)
plt.plot(k_values, error_rate, marker='o', linestyle='--', color='b')
plt.title('Error Rate vs. K Value')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.xticks(k_values)

# Specificity Plot
plt.subplot(1, 2, 2)
plt.plot(k_values, specificity_values, marker='o', linestyle='--', color='g')
plt.title('Specificity vs. K Value')
plt.xlabel('K Value')
plt.ylabel('Specificity')
plt.xticks(k_values)

plt.tight_layout()
plt.show()

# Best k-value
best_k = k_range[np.argmax(k_scores)]
print(f"Optimal Number of Neighbors (k): {best_k}")

# Refit the model with the best k-value
optimal_knn = KNeighborsClassifier(n_neighbors=best_k)
optimal_knn.fit(X_train, y_train)
optimal_y_pred = optimal_knn.predict(X_test)

#calculating AUC-ROC on train set
from sklearn.metrics import roc_curve, auc
#calculating AUC-ROC on test set

test_fpr, test_tpr, test_thresholds = roc_curve(y_test, knn_model.predict_proba(X_test_scaled)[:, 1])
test_roc_auc = auc(test_fpr, test_tpr)
print(test_roc_auc)

#visualizing AUC-ROC on test set
plt.figure()
plt.plot(test_fpr, test_tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % test_roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

#setting the value k or neighbors at 20 to choose one of the goodfit model
knn_model_20 = KNeighborsClassifier(20)
#training the knn model with train data
knn_model_20.fit(X_train_scaled, y_train)

test_fpr_20, test_tpr_20, test_thresholds_20 = roc_curve(y_test, knn_model_20.predict_proba(X_test_scaled)[:, 1])
test_roc_auc_20 = auc(test_fpr_20, test_tpr_20)

#visualizing AUC-ROC on test set
plt.figure()
plt.plot(test_fpr, test_tpr, color='darkorange', lw=2, label='5N ROC curve (area = %0.2f)' % test_roc_auc)
plt.plot(test_fpr_20, test_tpr_20, color='blue', lw=2, label='20N ROC curve (area = %0.2f)' % test_roc_auc_20)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


###########################################################################

