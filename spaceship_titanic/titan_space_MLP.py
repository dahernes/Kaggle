# %% [markdown]
# Categorical Analysis
# Spaceship Titanic
# Random Forest

# %%
# bibs
from pathlib import Path
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as torch_optim
from torchvision import models

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
# number of numerical columns and keys
embedded_col_names = embedded_cols.keys()
n_num_cols = len(X.columns) - 4

# %%
# embedding sizes
embedding_sizes = []
for _, n_categories in embedded_cols.items():
    size = (n_categories, min(50, (n_categories + 1) // 2))
    embedding_sizes.append(size)


# %%
# create pytorch dataset
class SpaceTitanicDataset(Dataset):
    def __init__(self, X, y, emb_objects):
        X = X.copy()
        self.X1 = X.loc[:, embedded_col_names].copy().values.astype(np.int64)
        self.X2 = X.drop(columns=embedded_col_names).copy().values.astype(np.float32)
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.X1[item], self.X2[item], self.y[item]


# %%
# creating train / valid datasets
train_ds = SpaceTitanicDataset(X_train, y_train, emb_object_features)
valid_ds = SpaceTitanicDataset(X_valid, y_valid, emb_object_features)

# %% [markdown]
# ## TODO MLP Model


# %%
# device compatible
# In order to make use of a GPU if available, we'll have to move our data and model to it.
def get_default_device():
    # pick right device
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    # move tensor to device
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    # wrap a dataloader to move data to a device
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        # yield a batch of data
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        # number of batches
        return len(self.dl)


# %%
# load device
device = get_default_device()
print(device)


# %%
# Model
class SpaceTitanicModel(nn.Module):
    def __init__(self, embedding_sizes, n_count):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings)  # length of all embeddings
        self.n_emb, self.n_cont = n_emb, n_count
        self.lin1 = nn.Linear(self.n_emb + self.n_cont, 200)
        self.lin2 = nn.Linear(200, 70)
        self.lin3 = nn.Linear(70, 1)
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(200)
        self.bn3 = nn.BatchNorm1d(70)
        self.emb_drop = nn.Dropout(0.6)
        self.drops = nn.Dropout(0.3)

    def forward(self, x_cat, x_cont):
        x = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x2 = self.bn1(x_cont)
        x = torch.cat([x, x2], 1)
        x = f.relu(self.lin1(x))
        x = self.drops(x)
        x = self.bn2(x)
        x = f.relu(self.lin2(x))
        x = self.drops(x)
        x = self.bn3(x)
        x = self.lin3(x)
        return x


# %%
model = SpaceTitanicModel(embedding_sizes, n_num_cols)
to_device(model, device)


# %%
# Optimizer
def get_optimizer(model, lr=0.001, wd=0.0):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch_optim.Adam(parameters, lr=lr, weight_decay=wd)
    return optim


# %%
# Training function
def train_model(model, optim, train_dl):
    model.train()
    total = 0
    sum_loss = 0
    for x1, x2, y in train_dl:
        batch = y.shape[0]
        output = model(x1, x2)
        loss = f.cross_entropy(output, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += batch
        sum_loss += batch * (loss.item())
    return sum_loss/total


# %%
# valuation function
def val_loss(model, valid_dl):
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0
    for x1, x2, y in valid_dl:
        current_batch_size = y.shape[0]
        out = model(x1, x2)
        loss = f.cross_entropy(out, y)
        sum_loss += current_batch_size * (loss.item())
        total += current_batch_size
        pred = torch.max(out, 1)[1]
        correct += (pred == y).float().sum().item()
        print("valid loss %.3f and accuracy %.3f" % (sum_loss/total, correct/total))
        return sum_loss/total, correct/total


# %%
# train loop
def train_loop(model, epochs, lr=0.01, wd=0.0):
    optim = get_optimizer(model, lr=lr, wd=wd)
    for i in range(epochs):
        loss = train_model(model, optim, train_dl)
        print('training loss: ', loss)
        val_loss(model, valid_dl)


# %%
# ## TODO Training

# %%
# train / valid dataloader
batch_size = 128
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)

# %%
train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)

# %%
train_loop(model, epochs=10, lr=0.05, wd=0.00001)

# %%
i = 1
for x1, x2, y in train_dl:
    print('batch_num:', i)
    i += 1
    print(x1, x2, y)
