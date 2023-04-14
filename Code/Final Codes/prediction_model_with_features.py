import pandas as pd
import warnings
import re
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse, urlsplit, parse_qs
import tldextract
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

#  df = pd.read_csv(r'D:\Downloads\model_dataset.csv')

url = r'D:\Downloads\final_xgb_features.csv'
df = pd.read_csv(url)
df = df.drop(columns=['url'], axis=1)
# df = df.loc[:, ['PageRank', 'dns_record', 'search_tld' ,'status']]

# Map "legitimate" and "phishing" to binary numbers
df["status"] = df["status"].map({"legitimate": 0, "phishing": 1})

X = df.drop(columns=['status']) #independent variable columns
y = df['status'] #target/dependent variable column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123) #80:20 or 80:30 ratio standard. 80 training 20 testing 


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
#                             min_impurity_decrease=0.0, 
#                             max_leaf_nodes=36, 
#                             max_features=25,
#                             max_depth=24, 
#                             ccp_alpha=0.0,
#                             random_state=142)


model.fit(X_train, y_train)

model.score(X_test, y_test)

# view the feature scores
feature_scores = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

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

from sklearn.metrics import cohen_kappa_score

# Make predictions on the testing set
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)

# Store the metrics in a DataFrame
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Kappa'],
    'Score': [accuracy, precision, recall, f1, kappa]
})

# Set the "Metric" column as the index
metrics_df.set_index('Metric', inplace=True)

# Print the metrics as a formatted matrix
print(metrics_df.to_string())

# Model prediction
from extract_features import *
import re

url_regex = re.compile(
    r'^(?:http|ftp)s?://'  # scheme
    r'(?:[\w-]+\.)+[a-z]{2,}'  # TLD
    r'(?:/?|[/?]\S+)$', re.IGNORECASE)

while True:
    url = input("URL (type 'stop' to exit): ")
    if url.lower() == "stop":
        break
    elif url_regex.match(url):
        user_input = extract_features(url)
        prediction = model.predict(user_input)
        confidence = model.predict_proba(user_input)[0]
        if prediction == 0:
            if confidence[0] >= 0.5:
                print("Prediction: Legitimate")
            else:
                print("Prediction: Suspicious")
        else:
            if confidence[1] >= 0.65:
                print("Prediction: Phishing")
            else:
                print("Prediction: Suspicious")
        print(f"Confidence: Legitimate={confidence[0]:.2f}, Phishing={confidence[1]:.2f}")
    else:
        print('Enter full url!')