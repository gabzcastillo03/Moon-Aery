import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse, urlsplit, parse_qs
import tldextract
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
warnings.filterwarnings("ignore")

#  df = pd.read_csv(r'D:\Downloads\model_dataset.csv')

url = "https://raw.githubusercontent.com/gabzcastillo03/Moon-Aery/main/Dataset/Final/superduper_final_features.csv"
df = pd.read_csv(url)
df = df.drop(columns=['url'], axis=1)

# Map "legitimate" and "phishing" to binary numbers
df["status"] = df["status"].map({"legitimate": 0, "phishing": 1})

X = df.drop(columns=['status']) #independent variable columns
y = df['status'] #target/dependent variable column


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123) #80:20 or 80:30 ratio standard. 80 training 20 testing 


forest_model = RandomForestClassifier(n_estimators=133,
                            min_samples_split=6,
                            min_samples_leaf=9, 
                            min_impurity_decrease=0.0, 
                            max_leaf_nodes=36, 
                            max_features=25,
                            max_depth=24, 
                            ccp_alpha=0.0,
                            random_state=142)

forest_model.fit(X_train, y_train)

forest_model.score(X_test, y_test)

from sklearn.metrics import cohen_kappa_score

# Make predictions on the testing set
y_pred = forest_model.predict(X_test)

# Calculate Cohen's Kappa statistic
kappa = cohen_kappa_score(y_test, y_pred)

print('Kappa Statistics:', kappa)

# Model prediction
from extract_features import *

while True:
    url = input("URL (type 'stop' to exit): ")
    if url.lower() == "stop":
        break
    user_input = extract_features(url)
    prediction = forest_model.predict([user_input])
    confidence = forest_model.predict_proba([user_input])[0]
    if prediction == 0:
        print("Prediction: Legitimate")
    else:
        print("Prediction: Phishing")
    print(f"Confidence: Legitimate={confidence[0]:.2f}, Phishing={confidence[1]:.2f}")