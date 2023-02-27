# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 16:44:35 2023

@author: Daniel
"""

import pandas as pd

df = pd.read_csv(r'C:\Users\Daniel\Documents\gabby reviewer\Phishing Detection\Dataset\model_dataset.csv')

# df = df.dropna()

#ECDF
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# for feature in df.columns:
#     if feature == 'status':
#         continue
#     for status in ['legitimate', 'phishing']:
#         data = df[df['status'] == status][feature]
#         x = np.sort(data)
#         y = np.arange(1, len(x)+1) / len(x)
#         sns.lineplot(x, y, label=status)
#     plt.xlabel(feature)
#     plt.ylabel('ECDF')
#     plt.legend()
#     plt.show()

#Correlation Matrix
# Set the figure size
plt.figure(figsize=(48,24)) #width-height

# Calculate correlation matrix
corr_matrix = df.corr()

# Plot heatmap of correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1) #crest, coolwarm, magma, flare
plt.title("Correlation Matrix")
plt.show()

# Map "legitimate" and "phishing" to binary numbers
df["status"] = df["status"].map({"legitimate": 0, "phishing": 1})

X = df.drop(columns=['status']) #independent variable columns
y = df['status'] #target/dependent variable column

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123) #80:20 or 80:30 ratio standard. 80 training 20 testing 

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

model.score(X_test, y_test)

#K fold cross validation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

tprs = np.zeros((skf.n_splits, 100))
aucs = np.zeros(skf.n_splits)
mean_fpr = np.linspace(0, 1, 100)

for i, (train, test) in enumerate(skf.split(X, y)):
    prob = model.predict_proba(X.iloc[train])[:, 1]
    fpr, tpr, _ = roc_curve(y.iloc[train], prob)
    tprs[i] = np.interp(mean_fpr, fpr, tpr)
    aucs[i] = auc(fpr, tpr)

mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
plt.plot(mean_fpr, mean_tpr, color='magenta', label=r'Mean ROC (AUC = %0.2f)' % (mean_auc), lw=2, alpha=1)

for i in range(skf.n_splits):
    plt.plot(mean_fpr, tprs[i], lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i+1, aucs[i]))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()

# view the feature scores
feature_scores = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
sns.barplot(x=feature_scores, y=feature_scores.index)
plt.gcf().set_size_inches(80,40)
# Add labels to the graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
# Add title to the graph
plt.title("Visualizing Important Features")
# Visualize the graph
plt.show()

#Model prediction

from FeatureExtraction import extract_features
url = input("URL: ")
user_input = extract_features(url)
model.predict(user_input)