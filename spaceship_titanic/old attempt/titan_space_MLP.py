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
# create pytorch dataset
class SpaceTitanicDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = X
        self.y_data = y

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]


# %%
# creating train / valid datasets
train_ds = SpaceTitanicDataset(torch.FloatTensor(X_train.values), torch.FloatTensor(y_train))
valid_ds = SpaceTitanicDataset(torch.FloatTensor(X_valid.values), torch.FloatTensor(y_valid))

# %%
# dataloader
BATCH_SIZE = 128
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=True)

# %% [markdown]
# ## TODO MLP Model
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
# model
input_features = len(X_train.columns)
l1_features = 64
output_feature = 1


class TitanSpaceModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Layer
        self.layer_1 = nn.Linear(input_features, l1_features)
        self.layer_2 = nn.Linear(l1_features, l1_features)
        self.layer_out = nn.Linear(l1_features, output_feature)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.batchnorm1 = nn.BatchNorm1d(l1_features)
        self.batchnorm2 = nn.BatchNorm1d(l1_features)

        def forward(self, inputs):
            x = self.relu(self.layer_1(inputs))
            x = self.batchnorm1(x)
            x = self.relu(self.layer_2(x))
            x = self.batchnorm2(x)
            x = self.dropout(x)
            x = self.layer_out(x)
            return x


# %%
model = TitanSpaceModel()
model.to(device)

print(model)

# %%
# Loss
loss_function = nn.BCELoss()
# Optimizer
learning_rate = 0.001
optimizer = torch_optim.Adam(model.parameters(), lr=learning_rate)
epochs = 100

# %%
# training
model.train()
train_loss = []
for epoch in range(epochs):
    for xb, yb in train_dl:
        y_pred = model(xb)                              # Forward pass
        calc_loss = loss_function(y_pred, yb)           # Loss Calculation
        optimizer.zero_grad()                           # previous gradient clearing
        loss_function.backward()                        # Backpropagation
        optimizer.step()                                # updating parameter
    train_loss.append(loss_function.item())
print("epoch: {} | loss: {}| accuracy: {}".format(epoch, calc_loss, y_pred))
