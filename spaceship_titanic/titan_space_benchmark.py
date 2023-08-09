# %% [markdown]
# study of the influence of data cleaning and feature engineering on prediction using the spaceship titanic dataset
# as an example (categorical prediction)

# TODO Beschreibung des Vorgehens
# TODO Einleitung: Dataset spaceship titanic -> link + problembeschreibung
# TODO Einleitung: Model XGB -> Erklärung warum und durchführung mit gridsearch + CV

# Creating the Benchmark

# %%
# import modules
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

import xgboost as xgb

from pathlib import Path
import pandas as pd
import seaborn as sns
sns.set(style="darkgrid")

# %%
# data paths
base_folder = Path.cwd()
data_folder = base_folder / "data"

# %%
# load train dataset
titan = pd.read_csv(data_folder / "train.csv")

# %%
# train/valid data split
titan_benchmark = titan.copy()

x = titan_benchmark
y = x.pop("Transported").values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1337)

# %%
model = xgb.XGBClassifier()
pipeline = Pipeline([
    ('model', model)
])

# Parameter
param_grid = {
    'model__objective': ['binary:logistic'],
    'model__random_state': [1337],
    'model__use_label_encoder': [False],
    'model__max_depth': [7],
    'model__n_estimators': [500],
    'model__colsample_bytree': [1],
    'model__learning_rate': [0.1]
}

# Hyperparameter Optimierung via GridSearch
grid_xgb = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=2, return_train_score=True)

if __name__ == '__main__':
    grid_xgb.fit(x_train, y_train)

# %%
# Results DF
threshold_df = pd.DataFrame(grid_xgb.cv_results_).sort_values('rank_test_score')
cols = ['mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score',
        'param_model__colsample_bytree', 'param_model__learning_rate',
        'param_model__max_depth', 'param_model__n_estimators']
print(threshold_df[cols])

# %%
# train / valid results
acc_score_train = grid_xgb.score(x_train, y_train)
acc_score_xgbtitan_train = round(acc_score_train * 100, 2)
print(acc_score_xgbtitan_train)

xgb_train_predicts_test = grid_xgb.predict(x_test)
acc_test_xgb = accuracy_score(y_test, xgb_train_predicts_test)
print(round(acc_test_xgb * 100, 2))
