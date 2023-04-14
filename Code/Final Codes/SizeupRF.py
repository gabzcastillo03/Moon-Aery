import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import time
import xgboost as xgb
import matplotlib.pyplot as plt

# Load the data from a CSV file
df = pd.read_csv(r'D:\Downloads\final_xgb_features.csv')
df = df.drop(columns=['url'], axis=1)

df["status"] = df["status"].map({"legitimate": 0, "phishing": 1})


X = df.drop(columns=['status']) #independent variable columns
y = df['status']

# Define the list of sample sizes to test
sample_sizes = [5000, 10000, 20000, 40000]

# Define an empty list to store the runtimes
num_cores = [1, 2, 3, 4]

# Define an empty list to store the runtimes
runtimes = []

# Iterate over the sample sizes and number of cores and train/test the model for each
for size in sample_sizes:
    for n in num_cores:
    
        X_train, X_test, y_train, y_test = train_test_split(X[:size], y[:size], test_size=0.3, random_state=123)

        # Initialize the Random Forest model
        
        model = xgb.XGBClassifier(gamma=0.1, 
                          learning_rate=0.1, 
                          max_depth=5, 
                          min_child_weight=1, 
                          n_estimators=130, 
                          subsample=0.8,
                          reg_lambda=2,
                          colsample_bytree=0.8,
                          random_state=142,
                          n_jobs=n)
        
        # model = RandomForestClassifier(n_estimators=133,
        #                             min_samples_split=6,
        #                             min_samples_leaf=9, 
        #                             min_impurity_decrease=0.0, 
        #                             max_leaf_nodes=36, 
        #                             max_features=25,
        #                             max_depth=24, 
        #                             ccp_alpha=0.0,
        #                             random_state=142,
        #                             n_jobs=n)

        # Measure the runtime of the model on the training data
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()

        # Calculate the runtime and append to the list
        runtime = end_time - start_time
        runtimes.append(runtime)

# Reshape the runtimes list into a 2D array with shape (len(sample_sizes), len(num_cores))
runtimes = np.array(runtimes).reshape((len(sample_sizes), len(num_cores)))

# Compute the size-up for each combination of sample size and number of cores
size_ups = runtimes / runtimes[:, [0]]

# Plot the size-up against the sample sizes for each number of cores
for i, n in enumerate(num_cores):
    plt.plot(sample_sizes, size_ups[:, i], marker='o', label=f'{n} cores')

plt.xlabel('Sample Size')
plt.ylabel('Size-up')
plt.title('XGBoost Size-up vs Sample Size and Number of Cores')
plt.legend()
plt.show()