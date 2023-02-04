# %% [markdown]
# Categorical Analysis
# Spaceship Titanic
# Random Forest

# %%
# bibs
from pathlib import Path
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
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

# flags
APPROACH_1 = False
APPROACH_2 = False
APPROACH_3 = False
APPROACH_4 = True

# constants
RAND_SEED = 1337
VALID_SIZE = 0.2


# %%
# helper functions
def object_to_numerical(dataframe, feature):
    for xi in feature:
        impute_data = {}
        loop_array = len(dataframe[xi].unique())

        for j in range(loop_array):
            num_trans = range(loop_array)
            impute_data[dataframe[xi].unique()[j]] = num_trans[j] + 1

        print(impute_data)

        dataframe[xi] = dataframe[xi].fillna(0)
        dataframe[xi] = dataframe[xi].replace(impute_data)
        dataframe[xi] = dataframe[xi].astype(int)


def percentile_check(df: pd.DataFrame, key: str, percentile_high: float = 0.95, percentile_low: float = 0.05) -> None:
    q_high = df[key].quantile(percentile_high)
    q_low = df[key].quantile(percentile_low)
    print(f"Percentile check for the metric column '{key}':")
    print(
        f"{'Number above':<18} {q_high:>3n} "
        f"({percentile_high * 100:>6.2f}%-Quantil):{df[df[key] > q_high][key].describe()['count']:>10,}")
    print(
        f"{'Number below':<18} {q_low:>3n} "
        f"({percentile_low * 100:>6.2f}%-Quantil):{df[df[key] < q_low][key].describe()['count']:>10,}")


# %% [markdown]
# ## Data Preparation

# %%
# Date Preparation
# two approaches
titan_dp = pd.read_csv(data_folder / 'titan_df_cleaned.csv')

# %%
print(titan_dp.dtypes)

# %%
# Approach 1
# change string/object features into numerical features
# select dtypes to convert into a numerical feature
if APPROACH_1:
    object_cols = titan_dp.select_dtypes(include='object').columns
    object_to_numerical(titan_dp, object_cols)

    # Dataformat
    float_cols = titan_dp.select_dtypes(include='float64').columns
    titan_dp[float_cols] = titan_dp[float_cols].astype(int)

    bool_features = ['CryoSleep', 'VIP', 'Transported']
    titan_dp = titan_dp.replace({'CryoSleep': {True: 1, False: 0}})
    titan_dp = titan_dp.replace({'VIP': {True: 1, False: 0}})
    titan_dp = titan_dp.replace({'Transported': {True: 1, False: 0}})
    titan_dp[bool_features] = titan_dp[bool_features].astype(int)

# %%
# Approach 2
# OneHot - Encoding for string/Object features
if APPROACH_2:
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

# %%
# Approach 3
# OneHot - Encoding + MinMaxScaling
if APPROACH_3:
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

# %%
# Approach 4
# OneHot - Encoding + Outlier Elimination -> Extension of approach 2
if APPROACH_4:
    titan_describe = titan_dp.describe()
    percentile_check(titan_dp, 'Total_Billed', 0.99, 0.01)  # 14298 - 86
    percentile_check(titan_dp, 'VRDeck', 0.99, 0.01)  # 4808.6 - 86
    percentile_check(titan_dp, 'Spa', 0.99, 0.01)  # 4580.16 - 85
    percentile_check(titan_dp, 'FoodCourt', 0.99, 0.01)  # 5973.16 - 85
    percentile_check(titan_dp, 'ShoppingMall', 0.99, 0.01)  # 2293.68 - 85
    percentile_check(titan_dp, 'RoomService', 0.99, 0.01)  # 3034.32 - 85

# %%
if APPROACH_4:
    total_billed_highpercentile = titan_dp[titan_dp['Total_Billed'] > 14298]
    titan_dp = titan_dp.drop(total_billed_highpercentile.index, axis=0)

    VRDeck_highpercentile = titan_dp[titan_dp['VRDeck'] > 4808.6]
    titan_dp = titan_dp.drop(VRDeck_highpercentile.index, axis=0)

    Spa_highpercentile = titan_dp[titan_dp['Spa'] > 4580.16]
    titan_dp = titan_dp.drop(Spa_highpercentile.index, axis=0)

    FC_highpercentile = titan_dp[titan_dp['FoodCourt'] > 5973.16]
    titan_dp = titan_dp.drop(FC_highpercentile.index, axis=0)

    SM_highpercentile = titan_dp[titan_dp['ShoppingMall'] > 2293.68]
    titan_dp = titan_dp.drop(SM_highpercentile.index, axis=0)

    RS_highpercentile = titan_dp[titan_dp['RoomService'] > 3034.32]
    titan_dp = titan_dp.drop(RS_highpercentile.index, axis=0)

# %%
if APPROACH_4:
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
# ## Model

# %%
# Train/Test Data Split
titan = titan_dp.copy()
X = titan
y = X.pop('Transported').values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VALID_SIZE, random_state=RAND_SEED)

# %%
# TODO CatBoost

# %%
# Approach 1:
# mean Train_Score =
# mean Valid_Score =
#
# Approach 2:
# mean Train_Score =
# mean Valid_Score =
#
# Approach 3:
# mean Train_Score =
# mean Valid_Score =
#
# Approach 4:
# mean Train_Score =
# mean Valid_Score =
