# %% [markdown]
# Categorical Analysis
# Spaceship Titanic

# %%
# bibs
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

# %%
# global variables
base_folder = Path.cwd()
data_folder = base_folder / "data"

# %%
# load train data
titan_df = pd.read_csv(data_folder / 'train.csv')


# %%
# Helper Functions
def df_missing_and_unique(df: pd.DataFrame) -> pd.DataFrame:
    m_and_u = pd.DataFrame(df.isna().sum().sort_values(ascending=False), columns=['missing_count'])
    m_and_u['missing_share'] = m_and_u['missing_count'] / len(df)*100
    m_and_u['unique_count'] = df.nunique().sort_values(ascending=False)
    m_and_u['unique_share'] = m_and_u['unique_count'] / len(df)*100
    return m_and_u


def reference_impute(dataframe, target, reference):
    filtered_by_nan = dataframe[dataframe[target].isnull()]

    for i in list(filtered_by_nan.index):
        buffer_group = dataframe.loc[i, reference]
        filtered_by_group = dataframe[dataframe[reference] == buffer_group][target].fillna(0)
        for j in filtered_by_group:
            if j != 0:
                dataframe.loc[i, target] = j
                break


def create_mv_per_row(dataframe):
    dataframe['MV_counter'] = dataframe.apply(lambda x: x.count(), axis=1)
    dataframe['MV_counter'] = len(dataframe.columns) - dataframe['MV_counter']


# %% [markdown]
# data overview

# %%
# shape
print(titan_df.shape)

# %%
# info
print(titan_df.info())

# %% [markdown]
# ## EDA

# %%
# Transported - sum
transported_vc = titan_df['Transported'].value_counts()

sns.barplot(x=transported_vc.index.tolist(), y=transported_vc.values.tolist(), palette='Blues')
plt.title('Ratio Transported')

plt.show()

# %%
# Passengers / Groups
# PassengerId Split
passenger_split = titan_df['PassengerId'].str.split('_', expand=True)
passenger_split = passenger_split.rename(columns={0: 'Group', 1: 'Passenger'})

titan_df['Group'] = passenger_split['Group']
titan_df['Passenger'] = passenger_split['Passenger']
del titan_df['PassengerId']

# %%
group_sum = len(titan_df['Group'].unique())
passenger_sum = len(titan_df['Passenger'])

sns.barplot(x=['Group', 'Passenger'], y=[group_sum, passenger_sum], palette='Blues')
plt.title('Sum of Groups & Passenger')

plt.show()

# %%
# HomePlanet / Destination shares - transported
homeplanet_vc = titan_df['HomePlanet'].value_counts()
destination_vc = titan_df['Destination'].value_counts()

bar_hp_transported = titan_df.groupby('HomePlanet').agg('sum')['Transported'].values.tolist()
bar_hp = homeplanet_vc.values.tolist()
bar_hp_subtracted = pd.Series(bar_hp).subtract(bar_hp_transported)
names_hp = homeplanet_vc.index.tolist()

bar_dest_transported = titan_df.groupby('Destination').agg('sum')['Transported'].values.tolist()
bar_dest_transported_sorted = sorted(bar_dest_transported, reverse=True)
bar_dest = destination_vc.values.tolist()
bar_dest_subtracted = pd.Series(bar_dest).subtract(bar_dest_transported_sorted)
names_dest = destination_vc.index.tolist()

barWidth = 1
r = [0, 1, 2]

colors_homeplanet = ['tab:blue', 'tab:green', 'tab:cyan']
colors_destination = ['tab:gray', 'tab:olive', 'tab:orange']

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle('HomePlanet/Destination - Transported')
fig.tight_layout()

ax1.pie(bar_hp, labels=names_hp, labeldistance=1.15,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
        colors=colors_homeplanet)

ax2.pie(destination_vc.values.tolist(), labels=destination_vc.index.tolist(), labeldistance=1.15,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
        colors=colors_destination)

ax3.bar(r, bar_hp_subtracted, color='#7f6d5f', edgecolor='white', width=barWidth, label='Not Transported')
ax3.bar(r, bar_hp_transported, bottom=bar_hp_subtracted, color='#557f2d', edgecolor='white', width=barWidth,
        label='Transported')
ax3.set_xticks(r, names_hp, fontweight='bold')
ax3.set_xlabel("HomePlanet")

ax4.bar(r, bar_dest_subtracted, color='#7f6d5f', edgecolor='white', width=barWidth, label='Not Transported')
ax4.bar(r, bar_dest_transported_sorted, bottom=bar_dest_subtracted, color='#557f2d', edgecolor='white', width=barWidth,
        label='Transported')
ax4.set_xticks(r, names_dest, fontweight='bold')
ax4.set_xlabel("Destination")
ax4.legend()

plt.show()

# %%
# total billed / age - passenger - vip
titan_df['Total_Billed'] = titan_df.loc[:, ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Passenger/VIP - Age / Total Billed')
fig.tight_layout()

sns.boxplot(ax=axes[0], x=titan_df['VIP'], y=titan_df['Age'], palette='Greens')
sns.boxplot(ax=axes[1], x=titan_df['VIP'], y=titan_df['Total_Billed'], palette='Blues')

plt.show()

# %%
# VIP Ratio
vip_ratio_filter = titan_df['VIP'].value_counts()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('VIP - Ratio')

ax1.pie(vip_ratio_filter.values, labels=vip_ratio_filter.index, labeldistance=1.15)
sns.barplot(ax=ax2, x=vip_ratio_filter.index, y=vip_ratio_filter.values, palette='muted')

plt.show()

# %%
# VIP - HomePlanet and destination separated
vip_filter = titan_df[titan_df['VIP'] == True]
agg_vip_hp = vip_filter[['HomePlanet', 'VIP']].groupby('HomePlanet').sum()
agg_vip_dest = vip_filter[['Destination', 'VIP']].groupby('Destination').sum()

vip_europe_dest = vip_filter[vip_filter['HomePlanet'] == 'Europa']['Destination'].value_counts()
vip_mars_dest = vip_filter[vip_filter['HomePlanet'] == 'Mars']['Destination'].value_counts()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle('VIP - Detailed view Homeplanet / Destination')

sns.barplot(ax=ax1, x=agg_vip_hp.index.tolist(), y=agg_vip_hp['VIP'].tolist(), palette='muted')
ax1.set_title('VIP - Homeplanet')

sns.barplot(ax=ax2, x=agg_vip_dest.index.tolist(), y=agg_vip_dest['VIP'].tolist(), palette='muted')
ax2.set_title('VIP - Destination')

sns.barplot(ax=ax3, x=vip_europe_dest.index.tolist(), y=vip_europe_dest.values, palette='muted')
ax3.set_title('Europe - Destination')

sns.barplot(ax=ax4, x=vip_mars_dest.index.tolist(), y=vip_mars_dest.values, palette='muted')
ax4.set_title('Mars - Destination')

plt.show()

# %%
# non-VIP - HomePlanet and destination separated
non_vip_filter = titan_df[titan_df['VIP'] == False]
non_vip_filter.loc[:, 'VIP'] = 1
agg_non_vip_hp = non_vip_filter[['HomePlanet', 'VIP']].groupby('HomePlanet').sum()
agg_non_vip_dest = non_vip_filter[['Destination', 'VIP']].groupby('Destination').sum()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
fig.suptitle('Non - VIP - Homeplanet / Destination')

sns.barplot(ax=ax1, x=agg_non_vip_hp.index.tolist(), y=agg_non_vip_hp['VIP'].tolist(), palette='muted')
ax1.set_title('Non - VIP - Homeplanet')

sns.barplot(ax=ax2, x=agg_non_vip_dest.index.tolist(), y=agg_non_vip_dest['VIP'].tolist(), palette='muted')
ax2.set_title('Non - VIP - Destination')

plt.show()

non_vip_europe_dest = non_vip_filter[non_vip_filter['HomePlanet'] == 'Europa']['Destination'].value_counts()
non_vip_mars_dest = non_vip_filter[non_vip_filter['HomePlanet'] == 'Mars']['Destination'].value_counts()
non_vip_earth_dest = non_vip_filter[non_vip_filter['HomePlanet'] == 'Earth']['Destination'].value_counts()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 8))
fig.suptitle('Non - VIP - Detailed view Homeplanet / Destination')

sns.barplot(ax=ax1, x=non_vip_europe_dest.index.tolist(), y=non_vip_europe_dest.values, palette='muted')
ax1.set_title('Europe - Destination')

sns.barplot(ax=ax2, x=non_vip_mars_dest.index.tolist(), y=non_vip_mars_dest.values, palette='muted')
ax2.set_title('Mars - Destination')

sns.barplot(ax=ax3, x=non_vip_mars_dest.index.tolist(), y=non_vip_mars_dest.values, palette='muted')
ax3.set_title('Mars - Destination')

plt.show()

# %%
# Money spending depending on HomePlanet
homeplanet_totalbilled_filter = titan_df[titan_df['CryoSleep'] == False][['HomePlanet', 'Total_Billed']]
agg_hp_tb = homeplanet_totalbilled_filter.groupby('HomePlanet').agg(
    mean=('Total_Billed', np.mean),
    sum=('Total_Billed', np.sum),
    max=('Total_Billed', np.max)
)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
fig.suptitle('Total_Billed - Homeplanet')
fig.tight_layout()

sns.barplot(ax=ax1, x=agg_hp_tb.index.tolist(), y=agg_hp_tb['mean'], palette='muted')
ax1.set_title('Mean')
ax1.set(ylabel=None)

sns.barplot(ax=ax2, x=agg_hp_tb.index.tolist(), y=agg_hp_tb['sum'], palette='muted')
ax2.set_title('sum')
ax2.set(ylabel=None)

sns.barplot(ax=ax3, x=agg_hp_tb.index.tolist(), y=agg_hp_tb['max'], palette='muted')
ax3.set_title('max')
ax3.set(ylabel=None)

plt.show()

# %%
# Money spending depending on HomePlanet and VIP
filter_vip_billed = titan_df[(titan_df['VIP'] == True) & titan_df['Total_Billed'] > 0]

sns.boxplot(x='HomePlanet', y='Total_Billed', data=filter_vip_billed)
plt.show()

# %%
# Money spending depending on HomePlanet and Non-VIP
filter_non_vip_billed = titan_df[(titan_df['VIP'] == False) & titan_df['Total_Billed'] > 0]

sns.boxplot(x='HomePlanet', y='Total_Billed', data=filter_non_vip_billed)
plt.show()

# %%
d = {
    'Europa_VIP': filter_vip_billed[filter_vip_billed['HomePlanet'] == 'Europa']['Total_Billed'],
    'Mars_VIP': filter_vip_billed[filter_vip_billed['HomePlanet'] == 'Mars']['Total_Billed'],
    'Europa_Non_VIP': filter_non_vip_billed[filter_non_vip_billed['HomePlanet'] == 'Europa']['Total_Billed'],
    'Mars_Non_VIP': filter_non_vip_billed[filter_non_vip_billed['HomePlanet'] == 'Mars']['Total_Billed'],
    'Earth_Non_VIP': filter_non_vip_billed[filter_non_vip_billed['HomePlanet'] == 'Earth']['Total_Billed']
     }

pd.DataFrame(data=d).describe()

# %%
# CryoSleep - Transported
cryosleep_vc = titan_df['CryoSleep'].value_counts()
cryosleep_df_filter = titan_df[titan_df['CryoSleep'] == True]
cryosleep_df_filter_vc = cryosleep_df_filter['Transported'].value_counts()
non_cryosleep_df_filter = titan_df[titan_df['CryoSleep'] == False]
non_cryosleep_df_filter_vc = non_cryosleep_df_filter['Transported'].value_counts()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
fig.suptitle('Cryosleep - Ratio - total - transported')

sns.barplot(ax=ax1, x=cryosleep_vc.index.tolist(), y=cryosleep_vc.values.tolist(), palette='muted')
ax1.set_title('Ratio Total')

sns.barplot(ax=ax2, x=cryosleep_df_filter_vc.index.tolist(), y=cryosleep_df_filter_vc.values.tolist(), palette='Blues')
ax2.set_title('Transported in Cryosleep')

sns.barplot(ax=ax3, x=non_cryosleep_df_filter_vc.index.tolist(), y=non_cryosleep_df_filter_vc.values.tolist(),
            palette='Blues')
ax3.set_title('Transported in Non - Cryosleep')

plt.show()

# %%
# Cryosleep - transported - vip
cryosleep_vip_filter = titan_df[(titan_df['CryoSleep'] == True) & (titan_df['VIP'] == True)]
cryosleep_vip_filter_vc = cryosleep_vip_filter['Transported'].value_counts()

cryosleep_non_vip_filter = titan_df[(titan_df['CryoSleep'] == True) & (titan_df['VIP'] == False)]
cryosleep_non_vip_filter_vc = cryosleep_non_vip_filter['Transported'].value_counts()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
fig.suptitle('Cryosleep - transported - vip')

sns.barplot(ax=ax1, x=cryosleep_df_filter_vc.index.tolist(), y=cryosleep_df_filter_vc.values.tolist(), palette='Blues')
ax1.set_title('Transported in Cryosleep')

sns.barplot(ax=ax2, x=cryosleep_vip_filter_vc.index.tolist(), y=cryosleep_vip_filter_vc.values.tolist(),
            palette='Blues')
ax2.set_title('Transported in Cryosleep - VIP True')

sns.barplot(ax=ax3, x=cryosleep_non_vip_filter_vc.index.tolist(), y=cryosleep_non_vip_filter_vc.values.tolist(),
            palette='Blues')
ax3.set_title('Transported in Cryosleep - VIP False')

plt.show()

# %%
# CryoSleep - Destination
cryosleep_df_filter_dest = cryosleep_df_filter['Destination'].value_counts()
cryosleep_df_filter_hp = cryosleep_df_filter['HomePlanet'].value_counts()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('CryoSleep True - HomePlanet - Destination')

ax1.pie(cryosleep_df_filter_hp.values, labels=cryosleep_df_filter_hp.index, labeldistance=1.15)
ax1.set_title('Homeplanet')

ax2.pie(cryosleep_df_filter_dest.values, labels=cryosleep_df_filter_dest.index, labeldistance=1.15)
ax2.set_title('Destination')

plt.show()

# %%
# Cabins
cabin_split = titan_df['Cabin'].str.split('/', expand=True)
cabin_split = cabin_split.rename(columns={0: 'Cabin_Deck', 1: 'Cabin_Number', 2: 'Cabin_Side'})

titan_df['Cabin_Deck'] = cabin_split['Cabin_Deck']
titan_df['Cabin_Number'] = cabin_split['Cabin_Number']
titan_df['Cabin_Side'] = cabin_split['Cabin_Side']
del titan_df['Cabin']

# %%
# POV Decks / Side sum
cabin_deck_vc = titan_df['Cabin_Deck'].value_counts()
cabin_side_vc = titan_df['Cabin_Side'].value_counts()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Cabin - Decks / Side')

sns.barplot(ax=ax1, x=cabin_deck_vc.index.tolist(), y=cabin_deck_vc.values.tolist(), palette='muted')
ax1.set_title('Decks')

sns.barplot(ax=ax2, x=cabin_side_vc.index.tolist(), y=cabin_side_vc.values.tolist(), palette='Blues')
ax2.set_title('Side')

plt.show()

# %%
# Decks - Side - Transported
decks_transported_true = titan_df[titan_df['Transported'] == True]['Cabin_Deck'].value_counts()
decks_transported_false = titan_df[titan_df['Transported'] == False]['Cabin_Deck'].value_counts()

decks_cryo_true = titan_df[titan_df['CryoSleep'] == True]['Cabin_Deck'].value_counts()
decks_cryo_false = titan_df[titan_df['CryoSleep'] == False]['Cabin_Deck'].value_counts()

# %%
# Age
sns.histplot(data=titan_df, x='Age', bins=20, kde=True)
plt.show()

# %%
# Age - Total_Billed
feat_age = titan_df['Age'].fillna(0)
feat_total_billed = titan_df['Total_Billed'].fillna(0)

fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
plt.suptitle('Age / Total_Billed')

sns.regplot(ax=ax1, x=feat_age, y=feat_total_billed, fit_reg=False)
ax1.set_title('Age / Total_Billed')

plt.show()

# %% [markdown]
# ## Data Cleaning

# %%
# df copy
titan_dc = titan_df.copy()

# %%
# df missing value heatmap
sns.set(style="darkgrid")
plt.figure(figsize=(12, 8))
sns.heatmap(titan_dc.isna().transpose(),
            cmap=sns.color_palette('muted'),
            cbar_kws={'label': 'Overview Missing Data'})
plt.show()

# %%
df_missing_and_unique(titan_dc)

# %%
# customize data types part 1
titan_dc[["Group", "Passenger"]] = titan_dc[["Group", "Passenger"]].astype(int)

# %%
# # Impute VIP through Homeplanet/Destination
# if Homeplanet = Earth -> VIP = False
titan_dc.loc[titan_dc[titan_dc['HomePlanet'] == 'Earth'].index, 'VIP'] = False

# %%
# if vip and dest: 55 Cancri e -> Homeplanet = Europa
filter_vip_dest = titan_dc[(titan_dc['Destination'] == '55 Cancri e') & (titan_dc['VIP'] == True)]
filter_vip_dest_isnull = filter_vip_dest[filter_vip_dest['HomePlanet'].isnull()]
titan_dc.loc[filter_vip_dest_isnull.index, 'HomePlanet'] = 'Europa'

# %%
# impute the rest 88 values of VIP -> false
titan_dc.loc[titan_dc[titan_dc['VIP'].isnull()]['VIP'].index, 'VIP'] = False

# %%
# impute Rest HomePlanet through PassengerGroup
reference_impute(titan_dc, 'HomePlanet', 'Group')

# %%
# Impute Destination through Passenger Group
reference_impute(titan_dc, 'Destination', 'Group')

# %%
# Check and Impute = no money spent if in CryoSleep
filtered_by_cryo = titan_dc[titan_dc["CryoSleep"] == True]
impute_feats = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
titan_dc.loc[list(filtered_by_cryo.index), impute_feats] = 0

# %%
# impute the left MV of the "spent" features with 0
titan_dc[impute_feats] = titan_dc[impute_feats].fillna(0)

# %%
# impute Cabin features through PassengerGroup
reference_impute(titan_dc, 'Cabin_Deck', 'Group')
reference_impute(titan_dc, 'Cabin_Number', 'Group')
reference_impute(titan_dc, 'Cabin_Side', 'Group')

# %%
# Age Features
# impute age through total_billed / VIP
age_tb_vip_filter = titan_df[['Age', 'Total_Billed', 'VIP']]
age_tb_vip_filter_wo_zeroes = age_tb_vip_filter[age_tb_vip_filter['Total_Billed'] > 0]

bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
age_tb_vip_filter_wo_zeroes['Cut'] = pd.cut(age_tb_vip_filter_wo_zeroes['Age'], bins)

agg_age = age_tb_vip_filter_wo_zeroes.groupby(['Cut', 'VIP']).agg(
    size_age=('Age', np.size),
    mean=('Total_Billed', np.mean),
    std=('Total_Billed', np.std),
    max=('Total_Billed', np.max),
    min=('Total_Billed', np.min)
)

# %%
# Age imputation
nan_age_filtered = titan_dc[titan_dc['Age'].isna()]
nan_age_filtered_index = nan_age_filtered.index

for i in nan_age_filtered_index:
    if titan_dc.loc[i, 'VIP']:
        if (titan_dc.loc[i, 'Total_Billed'] >= 954) & (titan_dc.loc[i, 'Total_Billed'] < 4000):
            titan_dc.loc[i, 'Age'] = 20
        elif (titan_dc.loc[i, 'Total_Billed'] >= 4000) & (titan_dc.loc[i, 'Total_Billed'] < 5680):
            titan_dc.loc[i, 'Age'] = 35
        else:
            titan_dc.loc[i, 'Age'] = 50
    elif not titan_dc.loc[i, 'VIP']:
        if (titan_dc.loc[i, 'Total_Billed'] >= 2) & (titan_dc.loc[i, 'Total_Billed'] < 1635):
            titan_dc.loc[i, 'Age'] = 20
        elif (titan_dc.loc[i, 'Total_Billed'] >= 1635) & (titan_dc.loc[i, 'Total_Billed'] < 2900):
            titan_dc.loc[i, 'Age'] = 30
        elif (titan_dc.loc[i, 'Total_Billed'] >= 2900) & (titan_dc.loc[i, 'Total_Billed'] < 3450):
            titan_dc.loc[i, 'Age'] = 40
        else:
            titan_dc.loc[i, 'Age'] = 50

# %%
# CryoSleep Features
cryo_vip_transported = titan_dc[(titan_dc['VIP'] == True) & (titan_dc['Transported'] == True)]
titan_dc.loc[cryo_vip_transported['CryoSleep'].isna().index, 'CryoSleep'] = True

# %%
cryo_age_billed = titan_dc[(titan_dc['Age'] > 18) & (titan_dc['Total_Billed'] == 0)]
cryo_age_billed_nan = cryo_age_billed[cryo_age_billed['CryoSleep'].isna()]
titan_dc.loc[cryo_age_billed_nan.index, 'CryoSleep'] = True

# %%
# count rows with 2, 3, 4+ MV
# Value counts MV in row
create_mv_per_row(titan_dc)

# %%
titan_dc['MV_counter'].value_counts()

# %%
# drop features and rows
drop_high_mv = titan_dc[titan_dc['MV_counter'] == 4].index
titan_dc = titan_dc.drop(drop_high_mv, axis=0)

titan_dc = titan_dc.drop(['Name'], axis=1)
titan_dc = titan_dc.drop(['MV_counter'], axis=1)

# %%
# last steps for finishing the cleaning
titan_dc.loc[titan_dc[titan_dc['CryoSleep'].isna()].index, 'CryoSleep'] = False

# impute with zeros
open_features = ['HomePlanet', 'Destination', 'Cabin_Number', 'Cabin_Deck', 'Cabin_Side']
titan_dc[open_features] = titan_dc[open_features].fillna(0)

# %%
# export cleaned df into csv
titan_dc.to_csv(data_folder/'titan_df_cleaned.csv', index=False)
