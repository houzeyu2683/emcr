
##
##  Load packages.
import pandas
import os
import sklearn.preprocessing

##
##  Storage folder.
storage = './resource/kaggle/prototype/clean/'
os.makedirs(os.path.dirname(storage), exist_ok=True)

##
##  Read sheets.
path = [
    './resource/kaggle/prototype/card.csv', 
    './resource/kaggle/prototype/history.csv'
]
##  Read the table and set the all value with string type.
sheet = {os.path.basename(p).split(".")[0]: pandas.read_csv(p, dtype=str) for p in path}

##  Define the data type in the first.
##  Then handle `card` about infinity and missing value.
table = sheet.get('card').copy()
table.head()
table.isna().sum()
##  Handle 'first_active_month'.
variable = 'first_active_month'
series = table[variable].astype(str).copy()
series[series=='nan'] = "2017-09"
table[variable] = series
##  Handle 'card_id'.
variable = 'card_id'
series = table[variable].astype(str).copy()
table[variable] = series
##  Handle 'feature_1'.
variable = 'feature_1'
series = table[variable].astype(int).copy()
series = series - series.min()
table[variable] = series
##  Handle 'feature_2'.
variable = 'feature_2'
series = table[variable].astype(int).copy()
table[variable] = series
series = series - series.min()
##  Handle 'feature_3'.
variable = 'feature_3'
series = table[variable].astype(int).copy()
series = series - series.min()
table[variable] = series
##  Handle 'target'.
variable = 'target'
series = table[variable].astype(float).copy()
table[variable] = series
##  Handle 'source'.
variable = 'source'
series = table[variable].astype(str).copy()
table[variable] = series

##
##  Generate variable base on original information.
origin = 'first_active_month'
series = table[origin].astype(str).copy()
table['first_active_year_point'] = [i.split('-')[0] for i in series]
table['first_active_month_point'] = [i.split('-')[0] for i in series]

##
##  Convert character to code.
for c in ['first_active_month', 'first_active_year_point', 'first_active_month_point']:

    encoder = sklearn.preprocessing.LabelEncoder()
    _ = encoder.fit(table[c])
    table[c] = encoder.transform(table[c])
    continue

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

