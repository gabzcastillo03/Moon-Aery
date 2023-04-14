# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve, StratifiedKFold
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
path = r'D:\Downloads\final_xgb_features.csv'
df = pd.read_csv(path)
df = df.drop(columns=['url'])
# df = df.drop(columns=['host_entropy', 'count_underscore', 'count_space',
#                       'count_and','vowel_con_ratio','count_equal','path_extension',
#                       'punycode','count_comma','count_tilde','count_colon','tld_in_subdomain','tld_in_bad_position','count_dollar','port'])
# df.to_csv(r'D:\Downloads\final_xgb_features.csv', index = False)

df["status"] = df["status"].map({"legitimate": 0, "phishing": 1})

# Prepare the input features and target variable
X = df.drop(['status'], axis=1)
y = df['status']

# Create a classifier

model = xgb.XGBClassifier(gamma=0.1, 
                          learning_rate=0.1, 
                          max_depth=5, 
                          min_child_weight=1, 
                          n_estimators=120, 
                          subsample=0.8,
                          reg_lambda=2,
                          colsample_bytree=0.8)


# model = RandomForestClassifier(n_estimators=133,
#                             min_samples_split=6,
#                             min_samples_leaf=9, 
#                             max_leaf_nodes=36, 
#                             max_features=25,
#                             max_depth=24, 
#                             random_state=142)
    
# Define the range of sample sizes to plot
train_sizes = np.linspace(0.01, 1.0, 10)

# Calculate the learning curve scores
train_sizes_abs, train_scores, test_scores = learning_curve(model, X, y, train_sizes=train_sizes,
                                                            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                                            scoring='accuracy')

# Calculate the mean and standard deviation of the training and cross-validation scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot the learning curve
plt.plot(train_sizes_abs, train_mean, label='Training score')
plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.2)
plt.plot(train_sizes_abs, test_mean, label='Cross-validation score')
plt.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.2)
plt.title('Learning Curve')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend()
plt.show()