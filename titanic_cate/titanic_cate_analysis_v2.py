# %% [markdown]
# # Kaggle - Titanic Dataset Analysis & Forecast
# ### y = 'survived'

# %%
# # Setup

# %% Import Bibliotheken
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

import xgboost as xgb

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# %% 
# flags, constants and globals
# global variable
base_folder = Path.cwd()
data_folder = base_folder / "data"

# flags
VERBOSITY = True

# constants
RAND_SEED = 1337
VALID_SIZE = 0.2


# %%
# functions
def percentile_check(df: pd.DataFrame, key: str, percentile_high: float = 0.95, percentile_low: float = 0.05) -> None:
    q_high = df[key].quantile(percentile_high)
    q_low = df[key].quantile(percentile_low)
    print(f"Percentile check for the metric column '{key}':")
    print(
        f"{'Number above':<18} {q_high:>3n} ({percentile_high * 100:>6.2f}%-Quantil):{df[df[key] > q_high][key].describe()['count']:>10,}")
    print(
        f"{'Number below':<18} {q_low:>3n} ({percentile_low * 100:>6.2f}%-Quantil):{df[df[key] < q_low][key].describe()['count']:>10,}")


def verteilungsplot(df, cols, debug=False):
    """
    Diese Funktion berechnet für die Spalten 'cols'
    des pandas DataFrames 'df' die Schiefen der
    Verteilungen für verschiedene Fälle:
        - keine Transformation ('original')
        - Kubische Transformation ('power')
        - Quadratwurzel-Transformation ('sqrt')
        - Natürliche Logarithmus-Transformation('ln')
    
    Diese Verteilungen werden anschließend in
    einem matplotlib Plot visualisiert.
    """
    # Aufbau des Plots je Spalte
    if not isinstance(cols, list):
        cols = [cols]

    fig, axes = plt.subplots(nrows=len(cols), ncols=3, figsize=(14, 1.5 * len(cols)), constrained_layout=False,
                             squeeze=False)

    # return Variable der Schiefen
    ret_skews = {}

    for col, ax_row in zip(cols, axes):
        if debug:
            print(f"---------{col}")

        # Anwendung der Transformationen auf die jew. Spalte
        vals = df[~df[col].isna()][col]
        power_transformed = np.power(vals[vals >= 0].astype("float"), 1 / 3)
        square_root = np.sqrt(vals[vals >= 0].astype("float"))

        # Berechnung der Schiefen
        skews = {
            'original': vals.skew(),
            'power': power_transformed.skew(),
            'sqrt': square_root.skew(),
        }

        ret_skews[col] = skews

        if debug:
            print(f"{skews['original']:.2f}, {skews['power']:.2f}, {skews['sqrt']:.2f}")

        # Ausgabe der Plots
        vals.plot.kde(ax=ax_row[0])
        power_transformed.plot.kde(ax=ax_row[1])
        square_root.plot.kde(ax=ax_row[2])

        # Angabe der Schiefe im oberen rechten Rand
        ax_row[0].annotate(f"{skews['original']:.2f}", xy=(0.98, 0.95), xycoords='axes fraction', size=10, ha='right',
                           va='top')
        ax_row[1].annotate(f"{skews['power']:.2f}", xy=(0.98, 0.95), xycoords='axes fraction', size=10, ha='right',
                           va='top')
        ax_row[2].annotate(f"{skews['sqrt']:.2f}", xy=(0.98, 0.95), xycoords='axes fraction', size=10, ha='right',
                           va='top')

        # Label der Spaltennamen
        ax_row[0].annotate(col, xy=(0, 0.5), xytext=(-ax_row[0].yaxis.labelpad - 5, 0),
                           xycoords=ax_row[0].yaxis.label, textcoords='offset points',
                           size='large', ha='right', va='center')

        # Beschriftung der Achsen mit Min, Max und Mean
        for axl, m in zip(ax_row, [vals, power_transformed, square_root]):
            axl.set_ylabel("")
            axl.set_xticklabels([f'{m.min():.1f}', f'{m.mean():.1f}', f'{m.max():.1f}'])
            axl.set_yticklabels([])
            axl.set_xticks([m.min(), m.mean(), m.max()])
            axl.set_yticks([])

    axes[0, 0].set_title("no transform")
    axes[0, 1].set_title("power transform")
    axes[0, 2].set_title("square root")

    fig.tight_layout()
    fig.subplots_adjust(left=0.15, top=0.95)

    plt.show()

    return ret_skews


# %%
# load data
titan_df = pd.read_csv(data_folder / 'train.csv')


# %% [markdown]
# ## Part 1: Data cleaning / preparation / exploration

# %% [markdown]
# ### Quick Overview

# %%
# Train df head
titan_df.head()

# %% [markdown]
# **Information about the data**<br><br>
# PassengerID - Passenger's identification<br>
# Survived - Status survived/missing,dead<br> 
# Pclass - Ticket/Passenger class<br>
# Name - Name of passenger<br>
# Sex - Gender of passenger<br>
# Age - Age of passenger<br>
# SibSp - Number of siblings or spouse<br>
# Parch - Number of parents or child<br>
# Ticket - Ticket number<br>
# Fare - Ticket price<br> 
# Cabin - Cabin number<br> 
# Embarked - homeport (C=Cherbourg, Q=Queenstown, S=Southampton)

# %%
# df info
titan_df.info()

# %%
# df describe
titan_df.describe()

# %%
# df missing value heatmap
sns.set(style="darkgrid")
plt.figure(figsize=(12, 8))
sns.heatmap(titan_df.isna().transpose(),
            cmap=sns.color_palette('muted'),
            cbar_kws={'label': 'Overview Missing Data'})

# %% [markdown]
# ### Data Cleaning and Preparation

# %% [markdown]
# #### Data Cleaning - Numerical features
# PassengerId/Survived/Pclass are ok!
# Age/SibSp/Parch/Fare

# %%
# 'Fare' Feature
# Distplot
sns.distplot(titan_df['Fare'], kde=False, color='blue', bins=20)
plt.title('Fare Dist Plot')
plt.xlabel('Price')
plt.ylabel('Total')
plt.show()

# %%
# Box Plot
plt.figure(figsize=(4, 12))
sns.boxplot(data=titan_df['Fare'], palette="muted")
plt.title('Fare Box Plot')
plt.xlabel('Fare feature')
plt.ylabel('Prize')
plt.show()

# %%
# percentile check
percentile_check(titan_df, 'Fare', 0.99, 0.01)

# %%
# 'Fare' feature outlier
titan_df[titan_df['Fare'] < 4]

# %%
# group by for imputation
agg_fare = titan_df.groupby('Pclass')['Fare'].agg([np.size, np.mean, np.std, np.median, np.min, np.max])

# %%
# impute nulls
pcl_idx = agg_fare.index.to_list()
pcl_median = round(agg_fare['median'], 2).to_list()

for i in pcl_idx:
    aggmask = (titan_df['Fare'] == 0) & (titan_df['Pclass'] == i)
    titan_df['Fare'] = titan_df['Fare'].mask(aggmask, pcl_median[i - 1])

# %%
# the upper outlier is true and should not be deleted

# %%
# 'Age' Feature
titan_df['Age'].isna().value_counts()

# %%
# 'Age' group by for imputation
bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 70, 90, 100, 150, 200, 250, 300]
agg_age = agg_fare = titan_df.groupby(['Pclass', pd.cut(titan_df['Fare'], bins)])['Age'].agg(
    [np.size, np.mean, np.std, np.median, np.min, np.max])

# %%
# impute table
agg_age

# %%
# 'Age' imputation
titan_df['Age'] = titan_df['Age'].fillna(0)

# %%
for i in range(len(titan_df['Age'])):
    if titan_df.loc[i, 'Age'] == 0:
        titan_df.loc[i, 'Age'] = agg_age.loc[titan_df.loc[i, 'Pclass'], titan_df.loc[i, 'Fare']]['median']

# %%
# null value check
titan_df['Age'].isna().value_counts()

# %%
# 'Age' dist plot
sns.distplot(titan_df['Age'], kde=False, color='blue', bins=40)
plt.title('Age Dist Plot')
plt.xlabel('Age')
plt.ylabel('Total')
plt.show()

# %%
# 'SibSp' & 'Parch' Features
#
# 'SibSp*
sns.distplot(titan_df['SibSp'], kde=False, color='blue', bins=10)
plt.title('SibSp')
plt.xlabel('Number of siblings or spouse')
plt.ylabel('Total')
plt.show()

# %%
# percentile check
percentile_check(titan_df, 'SibSp', 0.99, 0.01)

# %%
# 'Parch'
sns.distplot(titan_df['Parch'], kde=False, color='blue', bins=10)
plt.title('Parch')
plt.xlabel('Number of parents or child')
plt.ylabel('Total')
plt.show()

# %%
# percentile check
percentile_check(titan_df, 'Parch', 0.99, 0.01)

# %%
# creating a 'total_family' feature
titan_df['total_family'] = titan_df['SibSp'] + titan_df['Parch']

# %% [markdown]
# #### Data Cleaning - Categorial/Object features

# %%
# 'Embarked' Feature
titan_df[titan_df['Embarked'].isna()]

# %%
# both passengers have the same ticket id=113572 and are the only ones with this id
titan_df[titan_df['Ticket'] == '113572']

# %%
# searching a match in 'name'
# Icard, Miss., Amelie
# Stone, Mrs. George Nelson (Martha Evelyn)
# https://www.encyclopedia-titanica.org/titanic-survivor/amelia-icard.html
# boared = Southampton
titan_df['Embarked'] = titan_df['Embarked'].mask(titan_df['Ticket'] == '113572', 'S')

# %%
# Bar Plot 'Embarked'
sns.set(style="darkgrid")
titan_df['Embarked'].value_counts().plot(kind='barh')

# %%
# 'Sex' Feature
titan_df['Sex'] = titan_df['Sex'].map({'male': 1, 'female': 0})

# %%
# Bar Plot 'Sex' - Male=1, Female=0
sns.set(style="darkgrid")
titan_df['Sex'].value_counts().plot(kind='barh')

# %%
# 'Ticket' Feature
# https://www.encyclopedia-titanica.org/community/threads/ticket-numbering-system.20348/
# I don't think we pull out much value in this feature
# So a simple cleanup on the ticket number is sufficient so the agents are removed from this feature
ticket_split = titan_df["Ticket"].str.rsplit(" ", n=1, expand=True)
ticket_split.columns = 'T0 T1'.split()
ticket_split = ticket_split.fillna(np.nan)
ticket_split['T1'] = ticket_split['T1'].fillna(ticket_split['T0'])
ticket_split['T1'] = ticket_split['T1'].str.replace('LINE', '0').str.strip()
ticket_split['T1'] = ticket_split['T1'].astype(int)
del ticket_split['T0']
ticket_split.columns = ['Ticketnumber']

# %%
# concat ticketnummer
titan_df = pd.concat([titan_df, ticket_split], axis=1)
del titan_df['Ticket']

# %%
# 'Name' Feature
# will be deleted, as it is no longer useful for the models
# in the Date are better features to use, like sex and passengerID
del titan_df['Name']

# %%
# 'Cabin' Feature
# https://www.encyclopedia-titanica.org/titanic-deckplans/
# The cabin values consist of the deck and the room number. Hypothesis: This characteristic 
# should have a high influence on the forecast, like the ticket class and also correlate.
# We should split this feature in number/deck and check nan
titan_df['Cabin'].isna().value_counts()

# %% [markdown]
# We need to we must impute the missing<br>
# https://titanic.fandom.com/wiki/First_Class_Staterooms<br>
# First Class - Decks A/B/C<br>
# Most of them on B(101)/C(134) - highest on Boatdeck(6) - A(36) lowest on D(49) and F(4)<br>
# Sum = 330 First Class rooms<br>
# price from £400 to £870<br>
# <br>
# https://titanic.fandom.com/wiki/Second_Class_Cabins<br>
# Second Class - Decks D(39)/E(65)/(F64)<br>
# Sum = 168 Rooms<br>
# 
# https://titanic.fandom.com/wiki/Third_Class_cabins<br>
# Third Class - Decks D/E/F/G<br>
#
# It is true that the lower the deck the higher the probability of drowning
#
# Imputation is vague because the rate of missing values is high. We use this 
# imputation for testing purposes to see if we can increase the accuracy.
#
# After Testing various imputations. i decided to drop this feature 

# %%
# save df without 'Cabin' Feature
# dropping the feature for a direct comparison
titan_df_wo_cabin = titan_df.copy()
del titan_df_wo_cabin['Cabin']

# %% [markdown]
# ### Data Visualization & Preparation

# %%
# OneHot Encoding to transform the categorial features
oh_enc = OneHotEncoder(handle_unknown='ignore')

# %%
# titan_df_wo_cabin - 'Embarked'
transformer_wo_cabin = make_column_transformer(
    (oh_enc, ['Embarked']),
    remainder='passthrough')

transformed_wo_cabin = transformer_wo_cabin.fit_transform(titan_df_wo_cabin)
titan_df_wo_cabin = pd.DataFrame(transformed_wo_cabin, columns=transformer_wo_cabin.get_feature_names())
titan_df_wo_cabin.columns = titan_df_wo_cabin.columns.str.replace('onehotencoder__x0', 'Embarked')

# %%
# Correlation Plots
corr = titan_df_wo_cabin.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.set(style="darkgrid")
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('titan_df_wo_cabin corr plot')

# %%
# consideration of the skew
# check without logn (lots of zeroes)
_ = verteilungsplot(titan_df_wo_cabin, list(titan_df_wo_cabin.columns))

# %%
# transformation based on this defined dictionary
transformation_dict = {
    'Embarked_C': None,
    'Embarked_Q': None,
    'Embarked_S': None,
    'PassengerId': None,
    'Survived': None,
    'Pclass': None,
    'Sex': None,
    'Age': None,
    'SibSp': np.sqrt,
    'Parch': np.sqrt,
    'Fare': np.sqrt,
    'total_family': np.sqrt,
    'Ticketnumber': np.sqrt
}

# %%
for col, func in transformation_dict.items():
    if func is None:
        if VERBOSITY:
            print(f"Spalte '{col}': keine Transformation")
        continue
    if isinstance(func, tuple):
        if VERBOSITY:
            print(f"Spalte '{col}': Transformation mit '{str(func[0])}' und Parametern '{str(func[1])}'")
        titan_df_wo_cabin[col] = func[0](titan_df_wo_cabin[col], func[1])
    else:
        if VERBOSITY:
            print(f"Spalte '{col}': Transformation mit '{str(func)}'")
        titan_df_wo_cabin[col] = func(titan_df_wo_cabin[col])
print("Done")

# %%
titan_df_wo_cabin.describe()

# %% [markdown]
# ## Part 2: Train and evaluate different Models (Log Regression / Random Forest / XGBoost)

# %% [markdown]
# #### 1) Logistic Regression

# %%
# df train/test split
X = titan_df_wo_cabin.copy()
y = X.pop('Survived').values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VALID_SIZE, random_state=RAND_SEED)

# %%
# Logistic Regression
model = LogisticRegression()
pipeline = Pipeline([
    ('model', model)
])

param_grid = [
    {
        'model': [LogisticRegression()],
        'model__tol': np.linspace(0.001, 1),
        'model__C': np.linspace(0.0001, 1.0),
        'model__random_state': [RAND_SEED],
        'model__max_iter': [100],
        'model__solver': ['newton-cg', 'lbfgs'],
        'model__penalty': ['l2', 'none']
    }
]

cv = KFold(n_splits=5, shuffle=True, random_state=RAND_SEED)
# cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=1)

grid_cv_logr = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring='accuracy', verbose=2)

if __name__ == '__main__':
    grid_cv_logr.fit(X_train, y_train)

# %%
# Best Result Log Reg
# {'memory': None,
#  'steps': [('model',
#    LogisticRegression(C=0.2245673469387755, random_state=1337, solver='newton-cg',
#                       tol=0.40875510204081633))],
#  'verbose': False,
#  'model': LogisticRegression(C=0.2245673469387755, random_state=1337, solver='newton-cg',
#                     tol=0.40875510204081633),
#  'model__C': 0.2245673469387755,
#  'model__class_weight': None,
#  'model__dual': False,
#  'model__fit_intercept': True,
#  'model__intercept_scaling': 1,
#  'model__l1_ratio': None,
#  'model__max_iter': 100,
#  'model__multi_class': 'auto',
#  'model__n_jobs': None,
#  'model__penalty': 'l2',
#  'model__random_state': 1337,
#  'model__solver': 'newton-cg',
#  'model__tol': 0.40875510204081633,
#  'model__verbose': 0,
#  'model__warm_start': False}

# %%
logr_grid_results = pd.DataFrame(grid_cv_logr.cv_results_).sort_values("rank_test_score")
display(logr_grid_results)

# %%
display(grid_cv_logr.best_estimator_.get_params())

# %%
# score
logr_titan_wo_cabin_score = grid_cv_logr.best_estimator_.score(X_train, y_train)
logr_titan_wo_cabin_score_final = round(logr_titan_wo_cabin_score * 100, 2)
display(logr_titan_wo_cabin_score_final)

# %%
# get coefficients and weigths
coef_df = pd.DataFrame(np.exp(grid_cv_logr.best_estimator_.named_steps['model'].coef_), columns=X.columns).T
coef_df.sort_values(by=0, ascending=False).rename(columns={0: "Coef"})

# %%
# test scores
logr_titan_wo_cabin_test = grid_cv_logr.best_estimator_.predict(X_test)
logr_titan_wo_cabin_test_score = accuracy_score(y_test, logr_titan_wo_cabin_test, normalize=True)
logr_titan_wo_cabin_test_score_final = round(logr_titan_wo_cabin_test_score * 100, 2)
display(logr_titan_wo_cabin_test_score_final)

# %% 
# Result DF Log Reg
logr_data = {'Model': ['Log Reg'],
             'Train_Result': [logr_titan_wo_cabin_score_final],
             'Test_Result': [logr_titan_wo_cabin_test_score_final]}
logr_results = pd.DataFrame(data=logr_data)

# %% [markdown]
# #### 2) Random Forest Regression

# %%
# df train/test split
titan_df_wo_cabin_rf = titan_df_wo_cabin.copy()
X = titan_df_wo_cabin_rf
y = X.pop('Survived').values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RAND_SEED)

# %% Modell aufbauen
model = RandomForestClassifier()
pipeline = Pipeline([
    ('model', model)
])

param_grid = {
    'model__max_features': ['sqrt', 'log2'],
    'model__max_depth': [7, 10, 12, 14],
    'model__n_estimators': [10, 50, 100, 500],
    'model__random_state': [RAND_SEED],
    'model__criterion': ['gini']
}

# Hyperparameter Optimierung via GridSearch
grid_rf = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=2, return_train_score=True)

if __name__ == '__main__':
    grid_rf.fit(X_train, y_train)

# %%
# # Best Result RF
# {'model__criterion': 'gini',
#  'model__max_depth': 14,
#  'model__max_features': 'sqrt',
#  'model__n_estimators': 100,
#  'model__random_state': 1337}

# %%
threshold_df = pd.DataFrame(grid_rf.cv_results_).sort_values('rank_test_score')
print(threshold_df)

# %%
acc_score_train = grid_rf.score(X_train, y_train)
acc_score_rftitan_train = round(acc_score_train * 100, 2)
print(acc_score_rftitan_train)

rftitan_test_preds = grid_rf.predict(X_test)
rftitan_test_acc = accuracy_score(y_test, rftitan_test_preds)
acc_score_rftitan_test = round(rftitan_test_acc * 100, 2)
print(acc_score_rftitan_test)

# %% 
# Result DF Random Forest
rf_data = {'Model': ['RF'],
           'Train_Result': [acc_score_rftitan_train],
           'Test_Result': [acc_score_rftitan_test]}
rf_results = pd.DataFrame(data=rf_data)

# %%
# Plot feature Importance
perm_importance = permutation_importance(grid_rf, X_test, y_test, n_jobs=-1, random_state=RAND_SEED)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(X_test.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.show()

# %% [markdown]
# #### 3) XGBoost

# %%
# df train/test split
titan_df_wo_cabin_rf = titan_df_wo_cabin.copy()
X = titan_df_wo_cabin_rf
y = X.pop('Survived').values

df_dmatrix = xgb.DMatrix(data=X, label=y)
# Aufsplittung des Datensatzes in einem Training und Test Datensatz
# 20 % des Datensatzes wird zum Test Datensatz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RAND_SEED)

# %% Modell aufbauen
model = xgb.XGBClassifier()
# Pipeline bietet zudem Möglichkeiten für das Preprocessing
# wir konvertieren die Daten manuell
pipeline = Pipeline([
    ('model', model)
])

# Parameter
param_grid = {
    'model__objective': ['binary:logistic'],
    'model__colsample_bytree': [0.3, 0.5, 0.7, 1],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.15],
    'model__max_depth': [7, 10, 12, 14],
    'model__n_estimators': [50, 100, 500],
    'model__booster': ['gbtree', 'gblinear', 'dart'],
    'model__eval_metric': ['logloss'],
    'model__random_state': [RAND_SEED]
}

# Hyperparameter Optimierung via GridSearch
grid_xgb = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=3,
                        return_train_score=True)

if __name__ == '__main__':
    grid_xgb.fit(X_train, y_train)

# %%
# best result xgb
# param_model__colsample_bytree: 0.5
# param_model__learning_rate: 0.05
# param_model__max_depth: 14
# param_model__n_estimators: 100
# param_model__booster: 'dart'

# %%
# best result
threshold_df = pd.DataFrame(grid_xgb.cv_results_).sort_values('rank_test_score')
cols = ['mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score',
        'param_model__colsample_bytree', 'param_model__learning_rate',
        'param_model__max_depth', 'param_model__n_estimators', 'param_model__booster']
print(threshold_df[cols])

# %%
# Feature Importance
xgb.plot_importance(grid_xgb.best_estimator_.named_steps['model'])
plt.rcParams['figure.figsize'] = [30, 60]
plt.show()

# %%
xgb_train_predicts_train = grid_xgb.best_estimator_.predict(X_train)
acc_train_xgb = accuracy_score(y_train, xgb_train_predicts_train)
print(round(acc_train_xgb*100, 2))

xgb_train_predicts_test = grid_xgb.best_estimator_.predict(X_test)
acc_test_xgb = accuracy_score(y_test, xgb_train_predicts_test)
print(round(acc_test_xgb*100, 2))

# %%
xgb_data = {'Model': ['XGB'],
            'Train_Result': [round(acc_train_xgb*100, 2)],
            'Test_Result': [round(acc_test_xgb*100, 2)]}
xgb_results = pd.DataFrame(data=xgb_data)

# %%
# Final Results
final_results = pd.concat([logr_results, rf_results, xgb_results], ignore_index=True)
print(final_results)

#      Model  Train_Result  Test_Result
# 0  Log Reg         80.62        77.09
# 1       RF         99.16        81.01
# 2      XGB         97.89        80.45


# %% [markdown]
# ## Part 3: Forecasting the Testfile

# %%
# load the testfile
titan_df_valid = pd.read_csv(data_folder / 'test.csv')

# %% [markdown]
# ### prepare the text.csv for the forcasting

# %%
# creating a 'total_family' feature
titan_df_valid['total_family'] = titan_df_valid['SibSp'] + titan_df_valid['Parch']

# %% transform 'Sex' feature
titan_df_valid['Sex'] = titan_df_valid['Sex'].map({'male': 1, 'female': 0})

# %%
# clean up 'ticket' - feature
# 'Ticket' Feature
ticket_split_valid = titan_df_valid["Ticket"].str.rsplit(" ", n=1, expand=True)
ticket_split_valid.columns = 'T0 T1'.split()
ticket_split_valid = ticket_split_valid.fillna(np.nan)
ticket_split_valid['T1'] = ticket_split_valid['T1'].fillna(ticket_split_valid['T0'])
ticket_split_valid['T1'] = ticket_split_valid['T1'].astype(int)
del ticket_split_valid['T0']
ticket_split_valid.columns = ['Ticketnumber']

# %%
# concat ticketnummer
titan_df_valid = pd.concat([titan_df_valid, ticket_split_valid], axis=1)
del titan_df_valid['Ticket']

# %%
# del columns
del titan_df_valid['Cabin']
del titan_df_valid['Name']

# %%
# transform NaN
titan_df_valid['Age'] = titan_df_valid['Age'].fillna(0)
titan_df_valid['Fare'] = titan_df_valid['Fare'].fillna(0)

# %%
# 'Embarked' OneHot Encoding
transformer_df_valid = make_column_transformer(
    (oh_enc, ['Embarked']),
    remainder='passthrough')

transformed_df_valid = transformer_df_valid.fit_transform(titan_df_valid)
titan_df_valid = pd.DataFrame(transformed_df_valid, columns=transformer_df_valid.get_feature_names())
titan_df_valid.columns = titan_df_valid.columns.str.replace('onehotencoder__x0', 'Embarked')

# %%
# consideration of the skew
# check without logn (lots of zeroes)
_ = verteilungsplot(titan_df_valid, list(titan_df_valid.columns))

# %%
# transformation based on this defined dictionary
transformation_dict = {
    'Embarked_C': None,
    'Embarked_Q': None,
    'Embarked_S': None,
    'PassengerId': None,
    'Pclass': None,
    'Sex': None,
    'Age': None,
    'SibSp': np.sqrt,
    'Parch': np.sqrt,
    'Fare': np.sqrt,
    'total_family': np.sqrt,
    'Ticketnumber': np.sqrt
}

# %%
for col, func in transformation_dict.items():
    if func is None:
        if VERBOSITY:
            print(f"Spalte '{col}': keine Transformation")
        continue
    if isinstance(func, tuple):
        if VERBOSITY:
            print(f"Spalte '{col}': Transformation mit '{str(func[0])}' und Parametern '{str(func[1])}'")
        titan_df_valid[col] = func[0](titan_df_valid[col], func[1])
    else:
        if VERBOSITY:
            print(f"Spalte '{col}': Transformation mit '{str(func)}'")
        titan_df_valid[col] = func(titan_df_valid[col])
print("Done")

# %%
# since RF provided the results, the forecasts are made on this model
rf_predictions = grid_rf.predict(titan_df_valid)

rf_predictions_titanic = pd.DataFrame({'PassengerId': titan_df_valid['PassengerId'], 'Survived': rf_predictions})
rf_predictions_titanic = rf_predictions_titanic.astype(int)
rf_predictions_titanic.to_csv(data_folder / 'submission_titanic_dahernes.csv', index=False)
print(rf_predictions_titanic)
