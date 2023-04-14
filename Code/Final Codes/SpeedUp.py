import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from time import time

# Read the CSV data
df = pd.read_csv(r'D:\Downloads\final_xgb_features.csv')
df = df.drop(columns=['url'], axis=1)

# Map "legitimate" and "phishing" to binary numbers
df["status"] = df["status"].map({"legitimate": 0, "phishing": 1})

X = df.drop(columns=['status']) #independent variable columns
y = df['status'] #target/dependent variable column


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Define a range of values for n_jobs to test
n_jobs_range = [1, 2, 3, 4]

# Define a list to store the runtime for each value of n_jobs
runtime_list = []

# Measure the runtime of Random Forest for each value of n_jobs
for n_jobs in n_jobs_range:
    # Create a Random Forest classifier with 100 trees and set the number of CPU cores to n_jobs
    
    model = xgb.XGBClassifier(gamma=0.1, 
                              learning_rate=0.1, 
                              max_depth=5, 
                              min_child_weight=1, 
                              n_estimators=130, 
                              subsample=0.8,
                              reg_lambda=2,
                              colsample_bytree=0.8)
    
    # model = RandomForestClassifier(n_estimators=133,
    #                             min_samples_split=6,
    #                             min_samples_leaf=9, 
    #                             min_impurity_decrease=0.0, 
    #                             max_leaf_nodes=36, 
    #                             max_features=25,
    #                             max_depth=24, 
    #                             ccp_alpha=0.0, n_jobs=n_jobs)

    # Measure the runtime of Random Forest
    start_time = time()
    model.fit(X, y)
    end_time = time()

    # Append the runtime to the list for this value of n_jobs
    runtime_list.append(end_time - start_time)

# Compute the speed-up relative to n_jobs=1
speedup = np.array(runtime_list[0]) / np.array(runtime_list)

# Plot the speed-up as a function of n_jobs
plt.plot(n_jobs_range, speedup, 'o-')
plt.xlabel('Number of CPU cores')
plt.ylabel('Speed-up')
plt.title('Speed-up of XGBoost')
plt.show()
