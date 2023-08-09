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

import xgboost as xgb

from warnings import filterwarnings, simplefilter
simplefilter(action='ignore', category=FutureWarning)
filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=UserWarning)

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
# XGB Modelling + Hyperparameter Optimization
# Overfitting:
# The first way is to directly control model complexity using max_depth, min_child_weight, and gamma parameters.
# The second way is to add randomness to make training robust to noise with subsample and colsample_bytree.
# We can also reduce stepsize eta. We must remember to increase num_round when we try this.
#
# the ratio of features used (i.e. columns used); colsample_bytree. Lower ratios avoid over-fitting.
# the ratio of the training instances used (i.e. rows used); subsample. Lower ratios avoid over-fitting.
# the maximum depth of a tree; max_depth. Lower values avoid over-fitting.
# the minimum loss reduction required to make a further split; gamma. Larger values avoid over-fitting.
# the learning rate of our GBM (i.e. how much we update our prediction with each successive tree); eta.
# Lower values avoid over-fitting.
# the minimum sum of instance weight needed in a leaf, in certain applications this relates directly to the minimum
# number of instances needed in a node; min_child_weight. Larger values avoid over-fitting.
model = xgb.XGBClassifier()
pipeline = Pipeline([
    ('model', model)
])

# Parameter
param_grid = {
    'model__objective': ['binary:logistic'],
    'model__random_state': [RAND_SEED],
    'model__verbosity': [0],
    'model__use_label_encoder': [False],
    'model__max_depth': [5, 6, 7, 8, 9, 10],
    'model__n_estimators': [100, 250, 500],
    'model__colsample_bytree': [1],
    'model__learning_rate': [0.1]
}

# Hyperparameter Optimierung via GridSearch
grid_xgb = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=2, return_train_score=True)

if __name__ == '__main__':
    grid_xgb.fit(X_train, y_train)

# TODO early stopping implementation

# %%
# Parameter
# param_grid = {
#     'model__objective': ['binary:logistic'],
#     'model__colsample_bytree': [0.6, 0.7, 0.8, 0.9],
#     'model__max_depth': [7, 8, 9],
#     'model__n_estimators': [500],
#     'model__random_state': [RAND_SEED],
#     'model__verbosity': [0],
#     'model__use_label_encoder': [False],
#     'model__learning_rate': [0.05, 0.1, 0.2],
#     'model__validate_parameters': [False],
#     'model__gamma': [3, 4, 5, 6],
#     'model__subsample': [0.5]
# }

# approach 1
# param_grid = {
#     'model__objective': ['binary:logistic'],
#     'model__random_state': [RAND_SEED],
#     'model__verbosity': [0],
#     'model__use_label_encoder': [False],
#     'model__max_depth': [6],
#     'model__n_estimators': [250],
#     'model__colsample_bytree': [1],
#     'model__learning_rate': [0.05],
#     'model__min_child_weight': [1]
# }

# approach 2
# param_grid = {
#     'model__objective': ['binary:logistic'],
#     'model__random_state': [RAND_SEED],
#     'model__verbosity': [0],
#     'model__use_label_encoder': [False],
#     'model__max_depth': [6],
#     'model__n_estimators': [100],
#     'model__colsample_bytree': [0.9],
#     'model__learning_rate': [0.3],
# }

# approach 3
# param_grid = {
#     'model__objective': ['binary:logistic'],
#     'model__random_state': [RAND_SEED],
#     'model__verbosity': [0],
#     'model__use_label_encoder': [False],
#     'model__max_depth': [5],
#     'model__n_estimators': [100],
#     'model__colsample_bytree': [1],
#     'model__learning_rate': [0.2]
# }

# approach 4
# param_grid = {
#     'model__objective': ['binary:logistic'],
#     'model__random_state': [RAND_SEED],
#     'model__verbosity': [0],
#     'model__use_label_encoder': [False],
#     'model__max_depth': [5, 6, 7, 8, 9, 10],
#     'model__n_estimators': [100, 250, 500],
#     'model__colsample_bytree': [1],
#     'model__learning_rate': [0.1]
# }

# %%
# Results DF
threshold_df = pd.DataFrame(grid_xgb.cv_results_).sort_values('rank_test_score')
cols = ['mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score',
        'param_model__colsample_bytree', 'param_model__learning_rate',
        'param_model__max_depth', 'param_model__n_estimators']
print(threshold_df[cols])

# %%
# train & valid results
# xgb_train_predicts_train = grid_xgb.predict(X_train)
# acc_train_xgb = accuracy_score(y_train, xgb_train_predicts_train)
# print(round(acc_train_xgb * 100, 2))
acc_score_train = grid_xgb.score(X_train, y_train)
acc_score_xgbtitan_train = round(acc_score_train * 100, 2)
print(acc_score_xgbtitan_train)

xgb_train_predicts_test = grid_xgb.predict(X_test)
acc_test_xgb = accuracy_score(y_test, xgb_train_predicts_test)
print(round(acc_test_xgb * 100, 2))

# %%
# Approach 1:
# mean Train_Score = 89.33
# mean Valid_Score = 81.12
#
# Approach 2:
# mean Train_Score = 93.9
# mean Valid_Score = 81.06
#
# Approach 3:
# mean Train_Score = 89.32
# mean Valid_Score = 81.0
#
# Approach 4:
# mean Train_Score = 86.38
# mean Valid_Score = 80.48
