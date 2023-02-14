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
# Preprocessing
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
y = X.pop('Transported_T').values

print('Labels: {}'.format(set(y)))
print('Zero count = {}, One count = {}'.format(len(y) - sum(y), sum(y)))

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=VALID_SIZE, random_state=RAND_SEED)

# %%
# embedding layer for pytorch dataset
# categorical embedding for columns having more than two values
# embedded_cols = {n: len(col.cat.categories) for n,col in X.items() if len(col.cat.categories) > 2} größer 2 features
# embedded_cols
emb_object_features = ['HomePlanet_T', 'Destination_T', 'Cabin_Deck_T', 'Cabin_Side_T']
embedded_cols = {}
for col_i in emb_object_features:
    embedded_cols[str(col_i)] = len(X[str(col_i)].unique())

# %%
# embedding sizes
embedding_sizes = []
for _, n_categories in embedded_cols.items():
    size = (n_categories, min(50, (n_categories + 1) // 2))
    embedding_sizes.append(size)


# %%
# create pytorch dataset
class SpaceTitanicDataset(dataset):
    def __int__(self, X, y, emb_object_features):
        X = X.copy()
        self.X1 = X.loc[:, emb_object_features].copy()
        self.X2 = X.drop(columnms=emb_object_features.copy())
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.X1[item], self.X2[item], self.y[item]


# creating train / valid datasets
train_ds = SpaceTitanicDataset(X_train, y_train, emb_object_features)
valid_ds = SpaceTitanicDataset(X_valid, y_valid, emb_object_features)

# %% [markdown]
# ## TODO MLP Model
# TODO device compatible
# TODO Model

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
