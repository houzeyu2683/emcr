
##
##  Load packages.
import pandas
import os

##
##  Storage folder.
storage = './resource/clean/'
os.makedirs(os.path.dirname(storage), exist_ok=True)

##
##  Read sheets.
path = [
    './resource/prototype/index.csv', 
    './resource/prototype/history.csv'
]
sheet = [(os.path.basename(p), pandas.read_csv(p, dtype=str)) for p in path]

##
##  Handle missing value of the index table.
table = sheet[0][1]
table.loc[table['first_active_month'].isna()] = table['first_active_month'].value_counts().idxmax()
table.to_csv(os.path.join(storage, 'index.csv'), index=False)
table.head()
del table

##
##  Handle missing value of the history table.
table = sheet[1][1]
handle = {}
handle['previous'] = table.isna().sum().copy()
character = [
    'category_3', 'merchant_id', 'most_recent_sales_range', 
    'most_recent_purchases_range', 'category_4'
]
number = [
    'category_2', 'merchant_group_id', 'numerical_1', 'numerical_2',
    'avg_sales_lag3', 'avg_purchases_lag3', 'active_months_lag3', 
    'avg_sales_lag6', 'avg_purchases_lag6', 'active_months_lag6', 
    'avg_sales_lag12', 'avg_purchases_lag12', 'active_months_lag12'
]
for i in character: table[i] = table[i].fillna("<NA>")
for i in number: table[i] = table[i].fillna(-1)
handle['following'] = table.isna().sum().copy()
table.to_csv(os.path.join(storage, 'history.csv'), index=False)
del table

