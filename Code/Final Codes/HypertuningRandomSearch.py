import pandas as pd
import warnings
warnings.filterwarnings("ignore")

url = "https://raw.githubusercontent.com/gabzcastillo03/Moon-Aery/main/Dataset/Final/with_dns_record.csv"
df = pd.read_csv(url)
df = df.drop(columns=['url'], axis=1)

df["status"] = df["status"].map({"legitimate": 0, "phishing": 1})

from sklearn.model_selection import train_test_split
X = df.drop(columns=['status']) #independent variable columns   
y = df['status'] #target/dependent variable column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y) #80:20 or 80:30 ratio standard. 80 training 20 testing 

import xgboost as xgb
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import RandomizedSearchCV

parameters = {
    #address overfitting
    # 'max_depth': list(range(10, 20, 1)),
    # 'min_samples_split': list(range(5, 10, 1)), # 
    # 'min_samples_leaf': list(range(2, 5, 1)),
    # 'max_features': ['sqrt', 10, 25],
    # 'max_leaf_nodes': list(range(10, 30, 5)),
    # 'min_impurity_decrease': [0.0, 0.01],
    # 'ccp_alpha': [0.0, 0.01], 
    # 'n_estimators': list(range(100, 150, 3))
    
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 500, 1000],
    'subsample': [0.5, 0.8, 1.0],
    'colsample_bytree': [0.5, 0.8, 1.0],
    'reg_alpha': [0.0, 0.1, 0.5],
    'reg_lambda': [0.0, 0.1, 0.5]
} 

model = xgb.XGBClassifier(random_state=42)
scorer = make_scorer(accuracy_score)

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=parameters,
    n_iter=500, # number of times it search randomly
    cv=10,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring=scorer
)

random_search.fit(X_train, y_train)

best_params = random_search.best_params_
print("Best hyperparameters:", best_params)

best_model = random_search.best_estimator_

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test set accuracy:", accuracy)

cv_results = pd.DataFrame(random_search.cv_results_)
cv_results.to_csv(r'D:\Downloads\random_search_results.csv', index=False)