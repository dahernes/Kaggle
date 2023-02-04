# %% [markdown]
# Categorical Analysis
# Spaceship Titanic
# Random Forest

# %%
# bibs
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import make_column_transformer

# from warnings import filterwarnings, simplefilter
# simplefilter(action='ignore', category=FutureWarning)
# filterwarnings("ignore", category=FutureWarning)
# filterwarnings("ignore", category=UserWarning)

# %%
# global variables
base_folder = Path.cwd()
data_folder = base_folder / "data"

# constants
RAND_SEED = 1337
VALID_SIZE = 0.2

# %% [markdown]
# ## Data Preparation

# %%
# Date Preparation
# two approaches
titan_dp = pd.read_csv(data_folder / 'titan_df_cleaned.csv')

# %%
print(titan_dp.dtypes)

# %%
# TODO Preprocessing

# %% [markdown]
# ## TODO MLP Model

# %%
# Train/Test Data Split
# titan = titan_dp.copy()
# X = titan
# y = X.pop('Transported').values
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VALID_SIZE, random_state=RAND_SEED)

# %%
# mean Train_Score =
# mean Valid_Score =
