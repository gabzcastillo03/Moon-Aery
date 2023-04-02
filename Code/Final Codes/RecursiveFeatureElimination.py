# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 19:49:39 2023

@author: Gabriel
"""

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

path = r'D:\Downloads\extracted_features.csv'
df = pd.read_csv(path)
df = df.drop(columns=['url'])
df["status"] = df["status"].map({"legitimate": 0, "phishing": 1})

X = df.drop(columns=['status']) #independent variable columns
y = df['status'] #target/dependent variable 

from sklearn.ensemble import RandomForestClassifier
forest_model = RandomForestClassifier(max_depth=15, max_features='auto', n_estimators=100)

from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

# Define the cross-validation method1
cv = StratifiedKFold(n_splits=10)

# Create the RFECV object
rfecv = RFECV(estimator=forest_model,
              step=1,
              cv=cv,
              scoring='accuracy',
              verbose=1,
              n_jobs=-1)

# Fit the RFECV object to the data
rfecv.fit(X, y)

# Print the selected features
selected_features = X.columns[rfecv.support_]
print('Selected Features: ', selected_features)

X = df.drop(columns=['status']) #independent variable columns
y = df['status'] #target/dependent variable 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)
forest_model.fit(X_train, y_train)
forest_model.score(X_test, y_test)

# view the feature scores
feature_scores = pd.Series(forest_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Create horizontal bar chart
sns.set_style('whitegrid')
plt.figure(figsize=(12, 20))
sns.barplot(x=feature_scores, y=feature_scores.index, palette='magma')

# Add labels to the graph
plt.xlabel('Feature Importance Score', fontsize=16)
plt.ylabel('Features', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Add title to the graph
plt.title('Feature Importance', fontsize=20, weight='bold')

# Add annotations with feature importance scores
for i, v in enumerate(feature_scores):
    plt.text(v + 0.005, i, str(round(v, 4)), color='black', fontsize=12, ha='left', va='center')

# Show the graph
plt.show()















"""

plt.figure(figsize=(48,24))
corr_matrix = df.corr(method='pearson')

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Pearson Correlation Matrix")
plt.show()

plt.figure(figsize=(48,24))
corr_matrix = df.corr(method='spearman')

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Spearman Correlation Matrix")
plt.show()

plt.figure(figsize=(48,24))
corr_matrix = df.corr(method='kendall')

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Kendall Tau Correlation Matrix")
plt.show()

"""
























