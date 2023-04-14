import pandas as pd
import warnings
import xgboost as xgb
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings("ignore")

url = r'D:\Downloads\recent_initial_features.csv'
# https://raw.githubusercontent.com/gabzcastillo03/Moon-Aery/main/Dataset/Final/with_dns_record.csv
df = pd.read_csv(url)
df = df.drop(columns=['url','host_entropy', 'count_underscore', 'count_space',
                      'count_and','vowel_con_ratio','count_equal','path_extension',
                      'punycode','count_comma','count_tilde','count_colon','tld_in_subdomain','tld_in_bad_position','count_dollar','port'])

df["status"] = df["status"].map({"legitimate": 0, "phishing": 1})

X = df.drop(columns=['status']) #independent variable columns   
y = df['status'] #target/dependent variable column

start_time = time.time()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42) #80:20 or 80:30 ratio standard. 80 training 20 testing 

param_grid = {
    'n_estimators': [100, 110, 120],
    'learning_rate': [0.15, 0.2],
    'max_depth': [3, 5, 7],
    'gamma': [0.1, 0.2],
    'min_child_weight': [1, 2, 3],
    'subsample': [0.7, 0.8],
}

# Create a Random Forest Classifier object
model = xgb.XGBClassifier()
scorer = make_scorer(accuracy_score)


start_time = time.time()
# Create a GridSearchCV object with the parameter grid and the random forest classifier
grid_search = GridSearchCV(estimator=model, 
                           param_grid=param_grid, 
                           verbose=1, 
                           cv=10, 
                           n_jobs=-1,
                           scoring=scorer)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: ", grid_search.best_params_)

# Print the best score found by GridSearchCV
print("Best score: ", grid_search.best_score_)

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")

cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.to_csv(r'D:\Downloads\grid_search_results_update.csv', index=False)