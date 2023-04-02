from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve, StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the dataset
path = r'D:\Downloads\extracted_features.csv'
df = pd.read_csv(path)
df = df.drop(columns=['url'
                      # , 'prefix_suffix', 'shortening_service',
                      # 'having_ip_address', 'check_com', 'tld_in_path', 'tld_in_bad_position',
                      # 'tld_in_subdomain', 'count_colon',
                      # 'count_underscore', 'count_star', 'count_underscore',
                      # 'count_star', 'count_tilde', 'word_in_dict', 'hostname_dictionary_words',
                      # 'count_http_token', 'port', 'path_extension',
                      # 'punycode', 'count_at', 'count_dollar', 'count_exclamation',
                      # 'count_percentage', 'count_or', 'count_equal', 'count_and',
                      # 'count_space', 'count_semicolon', 'abnormal_subdomain', 'count_comma'
                      ], axis=1)
# df = df.loc[:, (df != 0).any(axis=0)]
# df["status"] = df["status"].map({"legitimate": 0, "phishing": 1})

# Prepare the input features and target variable
X = df.drop(['status'], axis=1)
y = df['status']

# Create a random forest classifier
rf = RandomForestClassifier(n_estimators=133,
                            min_samples_split=6,
                            min_samples_leaf=9, 
                            min_impurity_decrease=0.0, 
                            max_leaf_nodes=36, 
                            max_features=25,
                            max_depth=24, 
                            ccp_alpha=0.0,
                            random_state=142)
    
    
    # max_depth=11, max_features=25, 
    #                         n_estimators=148, min_impurity_decrease=0.0,
    #                         min_samples_split=7, min_samples_leaf=3, 
    #                         max_leaf_nodes=25, ccp_alpha=0.0)

# n_estimators=120, min_samples_split=9, max_depth=13,
#                              max_features='log2', criterion='entropy', 
#                             class_weight='balanced', bootstrap=True

# # {'n_estimators': 148, 'min_samples_split':
#     7, 'min_samples_leaf': 3, 'min_impurity_decrease': 0.0, 
#     'max_leaf_nodes': 25, 'max_features': 25, 'max_depth': 11, 
#     'ccp_alpha': 0.0}

# Define the range of sample sizes to plot
train_sizes = np.linspace(0.1, 1.0, 10)

# Calculate the learning curve scores
train_sizes_abs, train_scores, test_scores = learning_curve(rf, X, y, train_sizes=train_sizes,
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