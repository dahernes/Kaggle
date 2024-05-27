import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OrdinalEncoder

# read the data
df = pd.read_csv('train.csv')

# %%
df.info()

#%%
df.describe()

#%%
df.isnull().sum()

#%%
# missing values heatmap
plt.figure(figsize=(12, 6))
ax = plt.axes()
sns.heatmap(df.isna().transpose(), cbar=False, ax=ax)

plt.title("Missing Values")
plt.xlabel("MV")
plt.ylabel("Cols")

plt.show()

#%%
# data preprocessing
features = list(df.columns)
features.pop()

#%%
categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(df)
categorical_columns

#%%
# ordinal scaler
encoder = OrdinalEncoder().set_output(transform="pandas")
ordinal_encoded = encoder.fit_transform(df[categorical_columns])
ordinal_encoded

#%%
df = df.drop(columns=categorical_columns)

#%%
df = pd.concat([df, ordinal_encoded], axis=1)

#%%
# XGBoost Classifier
xgb_model = xgb.XGBClassifier()

# brute force scan for all parameters, here are the tricks
# usually max_depth is 6,7,8
# learning rate is around 0.05, but small changes may make big diff
# tuning min_child_weight subsample colsample_bytree can have
# much fun of fighting against overfit
# n_estimators is how many round of boosting
# finally, ensemble xgboost with multiple seeds may reduce variance
# 'nthread': [4],   # when use hyperthread, xgboost may become slower

parameters = {'objective': ['binary:logistic'],
              'learning_rate': [0.05, 0.01],  # so called `eta` value
              'max_depth': [6, 7, 8],
              'min_child_weight': [11],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [1000],  # number of trees, change it to 1000 for better results
              'missing': [-999],
              'seed': [1337]
              }

clf = GridSearchCV(xgb_model, parameters, n_jobs=5,
                   cv=StratifiedKFold(n_splits=5, shuffle=True),
                   scoring='roc_auc',
                   verbose=2, refit=True)

clf.fit(df[features], df["Transported"])

#%%
# trust your CV!
scores_df = clf.best_score_
params_df = clf.best_params_
print(round(scores_df * 100, 2))
print(params_df)
