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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

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
del titan_df_wo_cabin['PassengerId']

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
# z-score / standard scaler
scaled_columns = titan_df_wo_cabin.columns
scaled_columns_clean = list(scaled_columns)
del scaled_columns_clean[3]

# %%
titan_wo_survived = titan_df_wo_cabin.copy()
del titan_wo_survived['Survived']

# %%
scaled_features = StandardScaler().fit_transform(titan_wo_survived.values)
scaled_features_df = pd.DataFrame(scaled_features, index=titan_df_wo_cabin.index, columns=scaled_columns_clean)
features_df_full = pd.concat([scaled_features_df, titan_df_wo_cabin['Survived'].astype(int)], axis=1)

# %%
features_df_full.describe()

# %% [markdown]
# #### 3) Random Forest

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
    'model__max_features': ['sqrt', 'log2', None],
    'model__max_depth': [6, 8, 10, 12, 14],
    'model__n_estimators': [5, 7, 10, 50, 100, 500],
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
#  'model__max_depth': 12,
#  'model__max_features': 'log2',
#  'model__n_estimators': 500,
#  'model__random_state': 1337}

# {'model__criterion': 'gini', 'model__max_depth': 12, 'model__max_features': 'log2', 'model__n_estimators': 500, 
#  'model__random_state': 1337}
#
# grid_rf = RandomForestClassifier(criterion='gini', max_depth=8, max_features='log2', n_estimators=100,
#                                  random_state=RAND_SEED)
# grid_rf.fit(X_train, y_train)

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
# Plot feature Importance
perm_importance = permutation_importance(grid_rf, X_test, y_test, n_jobs=-1, random_state=RAND_SEED)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(X_test.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.show()


# %% [markdown]
# ## Part 3: Forecasting the Testfile

# %%
# load the testfile
titan_df_valid = pd.read_csv(data_folder / 'test.csv')
passengerid = titan_df_valid.copy()['PassengerId']

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
del titan_df_valid['PassengerId']

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
# scaling
scaled_features = StandardScaler().fit_transform(titan_df_valid.values)
scaled_features_df = pd.DataFrame(scaled_features, index=titan_df_valid.index, columns=titan_df_valid.columns)

# %%
# since RF provided the results, the forecasts are made on this model
rf_predictions = grid_rf.predict(scaled_features_df)

rf_predictions_titanic = pd.DataFrame({'PassengerId': passengerid, 'Survived': rf_predictions})
rf_predictions_titanic = rf_predictions_titanic.astype(int)
rf_predictions_titanic.to_csv(data_folder / 'submission_titanic_dahernes.csv', index=False)
print(rf_predictions_titanic)
