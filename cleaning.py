
##
##  Load packages.
import pandas
import numpy
import os
import sklearn.preprocessing

##
##  Storage folder.
storage = './resource/kaggle/restructuring/cleaning/'
os.makedirs(os.path.dirname(storage), exist_ok=True)

##
##  Read sheets.
resource = "./resource/kaggle/restructuring/"
path = [
    os.path.join(resource, 'card.csv'), 
    os.path.join(resource, 'history.csv')
]
##  Read the table and set the all value with string type.
sheet = {os.path.basename(p).split(".")[0]: pandas.read_csv(p, dtype=str) for p in path}

##  Handle `card`.
##  Convert the data type in the first,
##  if numeric, move to positive range.
##  Handle missing value, maybe fill value or other way.
table = sheet.get('card').copy()
##  Handle 'first_active_month'.
variable = 'first_active_month'
series = table[variable].astype("str").copy()
series[series=='nan'] = "2017-09"
table[variable] = series
##  Handle 'card_id'.
variable = 'card_id'
series = table[variable].astype("str").copy()
table[variable] = series
##  Handle 'feature_1'.
variable = 'feature_1'
series = table[variable].astype("int64").copy()
series = series - series.min()
table[variable] = series
##  Handle 'feature_2'.
variable = 'feature_2'
series = table[variable].astype('int64').copy()
table[variable] = series
series = series - series.min()
##  Handle 'feature_3'.
variable = 'feature_3'
series = table[variable].astype('int64').copy()
series = series - series.min()
table[variable] = series
##  Handle 'target'.
variable = 'target'
series = table[variable].astype('float64').copy()
table[variable] = series
##  Handle 'source'.
variable = 'source'
series = table[variable].astype("str").copy()
table[variable] = series

##
##  Generate variable base on origin.
origin = 'first_active_month'
series = table[origin].copy()
table['first_active_year_point'] = [i.split('-')[0] for i in series]
table['first_active_month_point'] = [i.split('-')[1] for i in series]

##
##  Category convert to code.
loop = ['first_active_month', 'first_active_year_point', 'first_active_month_point']
for c in loop:

    encoder = sklearn.preprocessing.LabelEncoder()
    _ = encoder.fit(table[c])
    table[c] = encoder.transform(table[c])
    continue

table.head()
key = ['card_id', 'target', 'source']
table[key].to_csv(os.path.join(storage, 'index.csv'), index=False)
key = ['card_id', 'first_active_month', 'feature_1', 'feature_2', 'feature_3']
table[key].to_csv(os.path.join(storage, 'card.csv'), index=False)

##
##  Handle `history`,
##  convert the data type.
##  if numeric, move to positive range.
##  Handle missing value, maybe fill value or other way.
table = sheet.get('history').copy()
##  Handle 'authorized_flag'.
variable = 'authorized_flag'
series = table[variable].astype("str").copy()
table[variable] = series
##  Handle 'card_id'.
variable = 'card_id'
series = table[variable].astype("str").copy()
table[variable] = series
##  Handle 'city_id'.
variable = 'city_id'
series = table[variable].astype("int64").copy()
series = series - series.min() 
table[variable] = series
##  Handle 'category_1'.
variable = 'category_1'
series = table[variable].astype("str").copy()
table[variable] = series
##  Handle 'installments'.
variable = 'installments'
series = table[variable].astype("int64").copy()
series = series - series.min()
table[variable] = series
##  Handle 'category_3'.
variable = 'category_3'
series = table[variable].astype("str").copy()
series[series=='nan'] = "-1"
table[variable] = series
##  Handle 'merchant_category_id'.
variable = 'merchant_category_id'
series = table[variable].astype("int").copy()
series = series - series.min()
table[variable] = series
##  Handle 'merchant_id'.
variable = 'merchant_id'
series = table[variable].astype("str").copy()
series[series=='nan'] = "-1"
table[variable] = series
##  Handle 'month_lag'.
variable = 'month_lag'
series = table[variable].astype("int64").copy()
series = series - series.min()
table[variable] = series
##  Handle 'purchase_amount'.
variable = 'purchase_amount'
series = table[variable].astype("float64").copy()
series = series - series.min()
table[variable] = series
##  Handle 'purchase_date'.
variable = 'purchase_date'
series = table[variable].astype("str").copy()
table[variable] = series
##  Handle 'category_2'.
variable = 'category_2'
series = table[variable].astype("float64").copy()
series = series - series.min()
series = series.fillna(-1)
table[variable] = series
##  Handle 'state_id'.
variable = 'state_id'
series = table[variable].astype("int64").copy()
series = series - series.min()
table[variable] = series
##  Handle 'subsector_id'.
variable = 'subsector_id'
series = table[variable].astype("int64").copy()
series = series - series.min()
table[variable] = series
##  Handle 'merchant_group_id'.
variable = 'merchant_group_id'
series = table[variable].astype('Int64').copy()
series = series - series.min()
series = series.fillna(-1)
table[variable] = series
##  Handle 'numerical_1'.
variable = 'numerical_1'
series = table[variable].astype('float64').copy()
series = series - series.min()
series = series.fillna(-1)
table[variable] = series
##  Handle 'numerical_2'.
variable = 'numerical_2'
series = table[variable].astype('float64').copy()
series = series - series.min()
series = series.fillna(-1)
table[variable] = series
##  Handle 'most_recent_sales_range'.
variable = 'most_recent_sales_range'
series = table[variable].astype('str').copy()
series[series=='nan'] = "-1"
table[variable] = series
##  Handle 'most_recent_purchases_range'.
variable = 'most_recent_purchases_range'
series = table[variable].astype('str').copy()
series[series=='nan'] = "-1"
table[variable] = series
##  Handle 'avg_sales_lag3'.
variable = 'avg_sales_lag3'
series = table[variable].astype('float64').copy()
series = series - series.min()
series = series.fillna(-1)
table[variable] = series
##  Handle 'avg_purchases_lag3'.
variable = 'avg_purchases_lag3'
series = table[variable].astype('float64').copy()
series = series - series.min()
ceiling = series[series!=numpy.inf].max() + series[series!=numpy.inf].std()
series[series==numpy.inf] = ceiling
series = series.fillna(-1)
table[variable] = series
##  Handle 'active_months_lag3'.
variable = 'active_months_lag3'
series = table[variable].astype('Int64').copy()
series = series - series.min()
series = series.fillna(-1)
table[variable] = series
##  Handle 'avg_sales_lag6'.
variable = 'avg_sales_lag6'
series = table[variable].astype('float64').copy()
series = series - series.min()
series = series.fillna(-1)
table[variable] = series
##  Handle 'avg_purchases_lag6'.
variable = 'avg_purchases_lag6'
series = table[variable].astype('float64').copy()
series = series - series.min()
ceiling = series[series!=numpy.inf].max() + series[series!=numpy.inf].std()
series[series==numpy.inf] = ceiling
series = series.fillna(-1)
table[variable] = series
##  Handle 'active_months_lag6'.
variable = 'active_months_lag6'
series = table[variable].astype('Int64').copy()
series = series - series.min()
series = series.fillna(-1)
table[variable] = series
##  Handle 'avg_sales_lag12'.
variable = 'avg_sales_lag12'
series = table[variable].astype('float64').copy()
series = series - series.min()
series = series.fillna(-1)
table[variable] = series
##  Handle 'avg_purchases_lag12'.
variable = 'avg_purchases_lag12'
series = table[variable].astype('float64').copy()
series = series - series.min()
ceiling = series[series!=numpy.inf].max() + series[series!=numpy.inf].std()
series[series==numpy.inf] = ceiling
series = series.fillna(-1)
table[variable] = series
##  Handle 'active_months_lag12'.
variable = 'active_months_lag12'
series = table[variable].astype('Int64').copy()
series = series - series.min()
series = series.fillna(-1)
table[variable] = series
##  Handle 'category_4'.
variable = 'category_4'
series = table[variable].astype('str').copy()
series[series=='nan'] = "-1"
table[variable] = series

##
##  Generate variable base on original.
origin = 'purchase_date'
series = table[origin].copy()
table['purchase_date_ymd'] = [i.split(" ")[0] for i in series]
table['purchase_date_ym'] = ["-".join(i.split(" ")[0].split('-')[0:2]) for i in series]
table['purchase_date_y'] = ["-".join(i.split(" ")[0].split('-')[0:1]) for i in series]

##
##  Convert category to code.
loop = [
    'authorized_flag', 'category_1', 'category_3', 'merchant_id',
    'most_recent_sales_range', 'most_recent_purchases_range', 'category_4',
    'purchase_date_ymd', 'purchase_date_ym', 'purchase_date_y'
]
for c in loop:

    encoder = sklearn.preprocessing.LabelEncoder()
    _ = encoder.fit(table[c])
    table[c] = encoder.transform(table[c])
    continue

table.head()
key = [
    'card_id', 'authorized_flag', 
    'city_id', 'category_1', 'installments', 
    'category_3', 'merchant_category_id', 
    'merchant_id', 'month_lag', 'purchase_amount', 
    'category_2', 'state_id', 'subsector_id',
    'merchant_group_id', 'numerical_1', 'numerical_2', 
    'most_recent_sales_range', 
    'most_recent_purchases_range',
    'avg_sales_lag3', 'avg_purchases_lag3', 
    'active_months_lag3', 'avg_sales_lag6', 
    'avg_purchases_lag6', 'active_months_lag6', 
    'avg_sales_lag12', 'avg_purchases_lag12',
    'active_months_lag12', 'category_4', 
    'purchase_date_ymd', 'purchase_date_ym',
    'purchase_date_y'
]
table[key].to_csv(os.path.join(storage, 'history.csv'), index=False)
