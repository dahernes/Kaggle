# %% [markdown]
# House Prize Regression

# %% Bibs
from pathlib import Path
import numpy as np
import pandas as pd

# %% flags, constants and globals
# global variable
base_folder = Path.cwd()
data_folder = base_folder / "data"

# %% functions

# %% train dataset
train_df = pd.read_csv(data_folder / 'train.csv')