# %% [markdown]
# ## House Prize Regression

# %%
# bibs
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb

# %% globals
# global variables
base_folder = Path.cwd()
data_folder = base_folder / "data"

# %%
# load train data
train_df = pd.read_csv(data_folder / 'train.csv')

# %% [markdown]
# Overview of data

# %%
print(train_df.shape)

# %%
print(train_df.info())

# %%
# df missing value heatmap
sns.set(style="darkgrid")
plt.figure(figsize=(12, 8))
sns.heatmap(train_df.isna().transpose(),
            cmap=sns.color_palette('muted'),
            cbar_kws={'label': 'Overview Missing Data'})
plt.show()

# %% [markdown]
# #### Model: XGBoost

# %%
train_df_xgb = train_df.copy()
X = train_df_xgb
y = X.pop('SalePrice').values

# %%
model_xgb = xgb.XGBRegressor()
model_xgb.fit(X, y)

# %%
model_xgb.score(X, y)
