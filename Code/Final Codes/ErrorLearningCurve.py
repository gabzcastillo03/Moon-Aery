# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve, StratifiedKFold
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'https://raw.githubusercontent.com/gabzcastillo03/Moon-Aery/main/Dataset/Final/final_xgb_features.csv'
df = pd.read_csv(path)
df = df.drop(columns=['url'], axis=1)
# df = df.loc[:, (df != 0).any(axis=0)]
df["status"] = df["status"].map({"legitimate": 0, "phishing": 1})

# Prepare the input features and target variable
X = df.drop(['status'], axis=1)
y = df['status']


model = xgb.XGBClassifier(gamma=0.1, 
                          learning_rate=0.1, 
                          max_depth=5, 
                          min_child_weight=1, 
                          n_estimators=120, 
                          subsample=0.8,
                          reg_lambda=2,
                          colsample_bytree=0.8)

train_sizes = np.linspace(0.1, 1.0, 10)

# Calculate the learning curve scores
train_sizes_abs, train_scores, test_scores = learning_curve(model, X, y, train_sizes=train_sizes,
                                                            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                                            scoring='accuracy')

# Train the classifier on the training data
train_error = 1 - train_scores
test_error = 1 - test_scores

# Calculate the mean and standard deviation of the error rates
train_mean = np.mean(train_error, axis=1)
train_std = np.std(train_error, axis=1)
test_mean = np.mean(test_error, axis=1)
test_std = np.std(test_error, axis=1)

# Plot the learning curve
plt.plot(train_sizes, train_mean, label="Training error")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
plt.plot(train_sizes, test_mean, label="Cross-validation error")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2)
plt.legend()
plt.xlabel("Training examples")
plt.ylabel("Error rate")
plt.show()
