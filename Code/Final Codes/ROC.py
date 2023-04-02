import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

df = pd.read_csv('D:\Downloads\extracted_features.csv')
df = df.drop(['url'], axis=1)
df = df.loc[:, (df != 0).any(axis=0)]

df["status"] = df["status"].map({"legitimate": 0, "phishing": 1})

X = df.drop(columns=['status']) #independent variable columns
y = df['status'] #target/dependent variable column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123) #80:20 or 80:30 ratio standard. 80 training 20 testing 


forest_model = RandomForestClassifier(n_estimators=109,
                            min_samples_split=6,
                            min_samples_leaf=3, 
                            min_impurity_decrease=0.0, 
                            max_leaf_nodes=25, 
                            max_features=25,
                            max_depth=17, 
                            ccp_alpha=0.0)
forest_model.fit(X_train, y_train)

forest_model.score(X_test, y_test)

skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

tprs = np.zeros((skf.n_splits, 100))
aucs = np.zeros(skf.n_splits)
mean_fpr = np.linspace(0, 1, 100)

for i, (train, test) in enumerate(skf.split(X, y)):
    prob = forest_model.predict_proba(X.iloc[train])[:, 1]
    fpr, tpr, _ = roc_curve(y.iloc[train], prob)
    tprs[i] = np.interp(mean_fpr, fpr, tpr)
    aucs[i] = auc(fpr, tpr)

mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
plt.plot(mean_fpr, mean_tpr, color='magenta', label=r'Mean ROC (AUC = %0.4f)' % (mean_auc), lw=2, alpha=1)

for i in range(skf.n_splits):
    plt.plot(mean_fpr, tprs[i], lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (i+1, aucs[i]))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()