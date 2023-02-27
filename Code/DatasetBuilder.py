# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 18:41:11 2023

@author: Daniel
"""

from FeatureExtraction import extract_features

github00 = "https://raw.githubusercontent.com/gabzcastillo03/Moon-Aery/main/Dataset/Clean/proper_legitimate.csv"
# github01 = 'https://raw.githubusercontent.com/gabzcastillo03/Moon-Aery/main/Dataset/Phishing/verified_online2.csv'
github01 = "https://raw.githubusercontent.com/gabzcastillo03/Moon-Aery/main/Dataset/Clean/proper_phishing.csv"

import pandas as pd

legitimate = pd.read_csv(github00)
phishing = pd.read_csv(github01)

# phishing = phishing[['url']]
# phishing = phishing.drop_duplicates()
# phishing  =  phishing.assign(status = 'phishing')

legitimate_count = legitimate.url.count()
phishing = phishing.sample(legitimate_count)

raw = pd.concat([legitimate, phishing], axis = 0)

df = pd.DataFrame(raw['url'].apply(extract_features).tolist())
df = pd.concat([df, raw['status'].reset_index(drop=True)], axis=1)

df.to_csv(r'C:\Users\Daniel\Documents\gabby reviewer\Phishing Detection\Dataset\model_dataset.csv', index = False)