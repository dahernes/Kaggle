# %% [markdown]
# Categorical Analysis
# Spaceship Titanic
# Random Forest

# %%
# bibs
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

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
titan_dp = pd.read_csv(data_folder / 'titan_df_cleaned.csv')

# %%
print(titan_dp.dtypes)

# %%
# TODO Preprocessing

# %%
# Normalization for numerical columns
float_cols = titan_dp.select_dtypes(include='float64').columns
int_cols = titan_dp.select_dtypes(include='int64').columns
numerical_cols = list(float_cols) + list(int_cols)
sorted_numerical_df = titan_dp[numerical_cols]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(sorted_numerical_df.values)
scaled_features_df = pd.DataFrame(scaled_features, index=titan_dp.index, columns=numerical_cols)

titan_dp = titan_dp.drop(numerical_cols, axis=1)
titan_dp = pd.concat([titan_dp, scaled_features_df], axis=1)

# %%
# Label encoding for categorical columns
object_cols = titan_dp.select_dtypes(include='object').columns
bool_cols = titan_dp.select_dtypes(include='bool').columns
categorical_cols = list(object_cols) + list(bool_cols)
sorted_categorical_cols = titan_dp[categorical_cols]

le = LabelEncoder()

for cate in categorical_cols:
    encoded_features = le.fit_transform(titan_dp[cate])
    titan_dp[cate + '_T'] = encoded_features

titan_dp = titan_dp.drop(categorical_cols, axis=1)

# %%
# Train / Valid Split
titan = titan_dp.copy()
X = titan
y = X.pop('Transported').values

print('Labels: {}'.format(set(y)))
print('Zero count = {}, One count = {}'.format(len(y) - sum(y), sum(y)))

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=VALID_SIZE, random_state=RAND_SEED)

# %%
# TODO embedding layer for pytorch dataset
# TODO create pytorch dataset

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
