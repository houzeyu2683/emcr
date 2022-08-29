
##
##  Load packages.
from queue import Empty
import pandas
import os
import sklearn.preprocessing
import itertools
import tqdm

##
##  Storage folder.
storage = './resource/kaggle/restructure/clean/engine/'
os.makedirs(os.path.dirname(storage), exist_ok=True)

##
##  Read sheets.
path = [
    './resource/kaggle/restructure/clean/card.csv', 
    './resource/kaggle/restructure/clean/history.csv'
]
##  Read the table and set the all value with default type.
sheet = {os.path.basename(p).split(".")[0]: pandas.read_csv(p) for p in path}

##
##  Combination feature,
##  The definition is base on primary key, 
##  group the category value to column, aggregate the numeric value.
def combine(table, index, category, numeric, aggregation='sum'):

    # table = sheet.get('history').copy()
    # index = 'card_id'
    # category = 'authorized_flag'
    # numeric = 'purchase_amount'
    # aggregation = 'sum'
    pivot = table.pivot_table(values=numeric, index=index, columns=category, aggfunc=aggregation, fill_value=0).copy()
    column = ["{}&{}:{}".format(category, i, numeric) for i in list(pivot.columns)]
    pivot.columns = column
    combination = pivot.reset_index()
    return(combination)

# table = sheet.get('history').head(200000).copy()
table = sheet.get('history').copy()
category = [
    'authorized_flag', 'city_id', 'category_1', 'installments',
    'category_3', 'merchant_category_id', 'month_lag', 
    'category_2', 'state_id', 'subsector_id',
    'most_recent_sales_range', 'active_months_lag3', 
    'active_months_lag6', 'active_months_lag12', 'category_4',
    'purchase_date_ymd', 'purchase_date_ym', 'purchase_date_y'
]
numeric = [
    'authorized_flag', 'city_id', 'category_1', 'installments',
    'category_3', 'merchant_category_id', 'merchant_id', 'month_lag', 
    'purchase_amount', 'category_2', 'state_id', 'subsector_id', 
    'merchant_group_id', 'numerical_1', 'numerical_2', 
    'most_recent_sales_range', 'most_recent_purchases_range',
    'avg_sales_lag3', 'avg_purchases_lag3', 'active_months_lag3', 
    'avg_sales_lag6', 'avg_purchases_lag6', 'active_months_lag6', 
    'avg_sales_lag12', 'avg_purchases_lag12', 'active_months_lag12', 
    'category_4', 'purchase_date_ymd', 'purchase_date_ym', 'purchase_date_y'
]
index = 'card_id'
loop = list(itertools.product(category, numeric))
for i, (c, n) in enumerate(tqdm.tqdm(loop)): 

    if(i==0): 
        
        group = pandas.DataFrame()
        batch = 0
        pass

    if(c!=n): 
        
        g = combine(table=table, index=index, category=c, numeric=n, aggregation='sum')
        if(group.empty): group = g
        else: group = pandas.merge(group, g, how='outer', on=index)
        pass
    
    dump = (group.memory_usage().sum() / 1024**3 > 10) if(not group.empty) else False
    if(dump):
        
        checkpoint = os.path.join(storage, 'general_feature_{}.csv'.format(batch))
        group.to_csv(checkpoint, index=False)
        group = pandas.DataFrame()
        batch = batch + 1

    continue


# each = 20
# for i, (c, n) in enumerate(tqdm.tqdm(loop)): 

    
#     if(i%each==0): group = pandas.DataFrame()
#     if(c!=n): 
        
#         g = combine(table=table, index=index, category=c, numeric=n, aggregation='sum')
#         if(group.empty): group = g
#         else: group = pandas.merge(group, g, how='outer', on=index)
#         pass

#     batch = (i+1)%each==0 or i+1==len(loop)
#     checkpoint = os.path.join(storage, 'general_feature_{}.csv'.format(i//each))
#     if(batch): 
        
#         group.to_csv(checkpoint, index=False)
#         # print(checkpoint)
#         # print(group.head())
#         # print(group.shape)

#     continue

# byte = group.memory_usage().sum()
# byte / 1024**3
# group.to_csv('./cache.csv')

# table.empty
# pandas.concat([group[0], group[1]], axis=1).keys()
# pandas.merge(group[0], group[1], how='outer', on='card_id')



# authorized_flag                       2
# card_id                          325540
# city_id                             308
# category_1                            2
# installments                         15
# category_3                            4
# merchant_category_id                331
# merchant_id                      334634
# month_lag                            16
# purchase_amount                  221246
# purchase_date                  17717516
# category_2                            6
# state_id                             25
# subsector_id                         41
# merchant_group_id                109389
# numerical_1                         955
# numerical_2                         948
# most_recent_sales_range               6
# most_recent_purchases_range           6
# avg_sales_lag3                     3368
# avg_purchases_lag3                99994
# active_months_lag3                    4
# avg_sales_lag6                     4504
# avg_purchases_lag6               135190
# active_months_lag6                    7
# avg_sales_lag12                    5006
# avg_purchases_lag12              172898
# active_months_lag12                  13
# category_4                            3
# purchase_date_ymd                   485
# purchase_date_ym                     16
# purchase_date_y                       2
# dtype: int64



# ##  Handle `card`, convert the data type in the first.
# ##  If numeric variable without target, move to positive range,
# ##  Handle missing value, maybe fill value or other way.
# table = sheet.get('card').copy()
# ##  Handle 'first_active_month'.
# variable = 'first_active_month'
# series = table[variable].astype("str").copy()
# series[series=='nan'] = "2017-09"
# table[variable] = series
# ##  Handle 'card_id'.
# variable = 'card_id'
# series = table[variable].astype("str").copy()
# table[variable] = series
# ##  Handle 'feature_1'.
# variable = 'feature_1'
# series = table[variable].astype("int64").copy()
# series = series - series.min()
# table[variable] = series
# ##  Handle 'feature_2'.
# variable = 'feature_2'
# series = table[variable].astype('int64').copy()
# table[variable] = series
# series = series - series.min()
# ##  Handle 'feature_3'.
# variable = 'feature_3'
# series = table[variable].astype('int64').copy()
# series = series - series.min()
# table[variable] = series
# ##  Handle 'target'.
# variable = 'target'
# series = table[variable].astype('float64').copy()
# table[variable] = series
# ##  Handle 'source'.
# variable = 'source'
# series = table[variable].astype("str").copy()
# table[variable] = series

# ##
# ##  Generate variable base on original information.
# origin = 'first_active_month'
# series = table[origin].copy()
# table['first_active_year_point'] = [i.split('-')[0] for i in series]
# table['first_active_month_point'] = [i.split('-')[1] for i in series]

# ##
# ##  Convert character to code.
# for c in ['first_active_month', 'first_active_year_point', 'first_active_month_point']:

#     encoder = sklearn.preprocessing.LabelEncoder()
#     _ = encoder.fit(table[c])
#     table[c] = encoder.transform(table[c])
#     continue

# table.head()
# table.to_csv(os.path.join(storage, 'card.csv'), index=False)
# del table

# ##
# ##  Handle `history`, convert the data type in the first.
# table = sheet.get('history').copy()
# ##  Handle 'authorized_flag'.
# variable = 'authorized_flag'
# series = table[variable].astype("str").copy()
# table[variable] = series
# ##  Handle 'card_id'.
# variable = 'card_id'
# series = table[variable].astype("str").copy()
# table[variable] = series
# ##  Handle 'city_id'.
# variable = 'city_id'
# series = table[variable].astype("int64").copy()
# series = series - series.min() 
# table[variable] = series
# ##  Handle 'category_1'.
# variable = 'category_1'
# series = table[variable].astype("str").copy()
# table[variable] = series
# ##  Handle 'installments'.
# variable = 'installments'
# series = table[variable].astype("int64").copy()
# series = series - series.min()
# table[variable] = series
# ##  Handle 'category_3'.
# variable = 'category_3'
# series = table[variable].astype("str").copy()
# series[series=='nan'] = "-1"
# table[variable] = series
# ##  Handle 'merchant_category_id'.
# variable = 'merchant_category_id'
# series = table[variable].astype("int").copy()
# series = series - series.min()
# table[variable] = series
# ##  Handle 'merchant_id'.
# variable = 'merchant_id'
# series = table[variable].astype("str").copy()
# series[series=='nan'] = "-1"
# table[variable] = series
# ##  Handle 'month_lag'.
# variable = 'month_lag'
# series = table[variable].astype("int64").copy()
# series = series - series.min()
# table[variable] = series
# ##  Handle 'purchase_amount'.
# variable = 'purchase_amount'
# series = table[variable].astype("float64").copy()
# series = series - series.min()
# table[variable] = series
# ##  Handle 'purchase_date'.
# variable = 'purchase_date'
# series = table[variable].astype("str").copy()
# table[variable] = series
# ##  Handle 'category_2'.
# variable = 'category_2'
# series = table[variable].astype("float64").copy()
# series = series - series.min()
# series = series.fillna(-1)
# table[variable] = series
# ##  Handle 'state_id'.
# variable = 'state_id'
# series = table[variable].astype("int64").copy()
# series = series - series.min()
# table[variable] = series
# ##  Handle 'subsector_id'.
# variable = 'subsector_id'
# series = table[variable].astype("int64").copy()
# series = series - series.min()
# table[variable] = series
# ##  Handle 'merchant_group_id'.
# variable = 'merchant_group_id'
# series = table[variable].astype('Int64').copy()
# series = series - series.min()
# series = series.fillna(-1)
# table[variable] = series
# ##  Handle 'numerical_1'.
# variable = 'numerical_1'
# series = table[variable].astype('float64').copy()
# series = series - series.min()
# series = series.fillna(-1)
# table[variable] = series
# ##  Handle 'numerical_2'.
# variable = 'numerical_2'
# series = table[variable].astype('float64').copy()
# series = series - series.min()
# series = series.fillna(-1)
# table[variable] = series
# ##  Handle 'most_recent_sales_range'.
# variable = 'most_recent_sales_range'
# series = table[variable].astype('str').copy()
# series[series=='nan'] = "-1"
# table[variable] = series
# ##  Handle 'most_recent_purchases_range'.
# variable = 'most_recent_purchases_range'
# series = table[variable].astype('str').copy()
# series[series=='nan'] = "-1"
# table[variable] = series
# ##  Handle 'avg_sales_lag3'.
# variable = 'avg_sales_lag3'
# series = table[variable].astype('float64').copy()
# series = series - series.min()
# series = series.fillna(-1)
# table[variable] = series
# ##  Handle 'avg_purchases_lag3'.
# variable = 'avg_purchases_lag3'
# series = table[variable].astype('float64').copy()
# series = series - series.min()
# series = series.fillna(-1)
# table[variable] = series
# ##  Handle 'active_months_lag3'.
# variable = 'active_months_lag3'
# series = table[variable].astype('Int64').copy()
# series = series - series.min()
# series = series.fillna(-1)
# table[variable] = series
# ##  Handle 'avg_sales_lag6'.
# variable = 'avg_sales_lag6'
# series = table[variable].astype('float64').copy()
# series = series - series.min()
# series = series.fillna(-1)
# table[variable] = series
# ##  Handle 'avg_purchases_lag6'.
# variable = 'avg_purchases_lag6'
# series = table[variable].astype('float64').copy()
# series = series - series.min()
# series = series.fillna(-1)
# table[variable] = series
# ##  Handle 'active_months_lag6'.
# variable = 'active_months_lag6'
# series = table[variable].astype('Int64').copy()
# series = series - series.min()
# series = series.fillna(-1)
# table[variable] = series
# ##  Handle 'avg_sales_lag12'.
# variable = 'avg_sales_lag12'
# series = table[variable].astype('float64').copy()
# series = series - series.min()
# series = series.fillna(-1)
# table[variable] = series
# ##  Handle 'avg_purchases_lag12'.
# variable = 'avg_purchases_lag12'
# series = table[variable].astype('float64').copy()
# series = series - series.min()
# series = series.fillna(-1)
# table[variable] = series
# ##  Handle 'active_months_lag12'.
# variable = 'active_months_lag12'
# series = table[variable].astype('Int64').copy()
# series = series - series.min()
# series = series.fillna(-1)
# table[variable] = series
# ##  Handle 'category_4'.
# variable = 'category_4'
# series = table[variable].astype('str').copy()
# series[series=='nan'] = "-1"
# table[variable] = series

# ##
# ##  Generate variable base on original information.
# origin = 'purchase_date'
# series = table[origin].copy()
# table['purchase_date_ymd'] = [i.split(" ")[0] for i in series]
# table['purchase_date_ym'] = ["-".join(i.split(" ")[0].split('-')[0:2]) for i in series]
# table['purchase_date_y'] = ["-".join(i.split(" ")[0].split('-')[0:1]) for i in series]

# ##
# ##  Convert character to code.
# loop = [
#     'authorized_flag', 'category_1', 'category_3', 'merchant_id', 'purchase_date',
#     'most_recent_sales_range', 'most_recent_purchases_range', 'category_4',
#     'purchase_date_ymd', 'purchase_date_ym', 'purchase_date_y'
# ]
# for c in loop:

#     encoder = sklearn.preprocessing.LabelEncoder()
#     _ = encoder.fit(table[c])
#     table[c] = encoder.transform(table[c])
#     continue

# table.head()
# table.to_csv(os.path.join(storage, 'history.csv'), index=False)
# del table
