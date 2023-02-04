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
# OneHot - Encoding + MinMaxScaling
float_cols = titan_dp.select_dtypes(include='float64').columns
int_cols = titan_dp.select_dtypes(include='int64').columns
numerical_cols = list(float_cols) + list(int_cols)
sorted_numerical_df = titan_dp[numerical_cols]

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(sorted_numerical_df.values)
scaled_features_df = pd.DataFrame(scaled_features, index=titan_dp.index, columns=numerical_cols)

titan_dp = titan_dp.drop(numerical_cols, axis=1)
titan_dp = pd.concat([titan_dp, scaled_features_df], axis=1)

oh_enc = OneHotEncoder(handle_unknown='ignore')
object_cols = titan_dp.select_dtypes(include='object').columns

for object_i in object_cols:
    transformer = make_column_transformer(
        (oh_enc, [object_i]),
        remainder='passthrough')

    transformed_array = transformer.fit_transform(titan_dp)
    titan_dp = pd.DataFrame(transformed_array, columns=transformer.get_feature_names())
    titan_dp.columns = titan_dp.columns.str.replace('onehotencoder__x0', object_i)

bool_features = ['CryoSleep', 'VIP', 'Transported']
titan_dp = titan_dp.replace({'CryoSleep': {True: 1, False: 0}})
titan_dp = titan_dp.replace({'VIP': {True: 1, False: 0}})
titan_dp = titan_dp.replace({'Transported': {True: 1, False: 0}})
titan_dp[bool_features] = titan_dp[bool_features].astype(int)
titan_dp = titan_dp.astype(float)

# %% [markdown]
# ## TODO MLP Model

# %%
# Train/Test Data Split
titan = titan_dp.copy()
X = titan
y = X.pop('Transported').values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VALID_SIZE, random_state=RAND_SEED)

# %%
# mean Train_Score =
# mean Valid_Score =
