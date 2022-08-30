
##
##  Load packages.
import pandas
import os
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
##  The definition is base on index, 
##  group the category value, aggregate the numeric value, compute combination.
def combine(table, index, category, numeric, aggregation='sum'):

    pivot = table.pivot_table(values=numeric, index=index, columns=category, aggfunc=aggregation, fill_value=0).copy()
    column = ["{}&{}:{}".format(category, c, numeric) for c in list(pivot.columns)]
    pivot.columns = column
    combination = pivot.reset_index()
    return(combination)

##
##  Assign category to numeric if unique number more than 100.
table = sheet.get('history').copy()
index = 'card_id'
category = [
    'authorized_flag', 'category_1', 'installments',
    'category_3', 'month_lag', 'category_2', 'state_id', 
    'subsector_id', 'most_recent_sales_range', 'most_recent_purchases_range',
    'active_months_lag3', 'active_months_lag6', 'active_months_lag12',
    'category_4', 'purchase_date_ym', 'purchase_date_y'
]
numeric = [
    'city_id', 'merchant_category_id', 'merchant_id',
    'purchase_amount', 'purchase_date', 'merchant_group_id',
    'numerical_1', 'numerical_2', 'avg_sales_lag3', 
    'avg_purchases_lag3', 'avg_sales_lag6', 'avg_purchases_lag6',
    'avg_sales_lag12', 'avg_purchases_lag12', 'purchase_date_ymd'
]
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
    if(dump or i==len(loop)-1):
        
        checkpoint = os.path.join(storage, 'combination_feature_{}.csv'.format(batch))
        group.to_csv(checkpoint, index=False)
        group = pandas.DataFrame()
        batch = batch + 1

    continue

del table

##
##  Statistic feature,
##  The definition is base on index, 
##  group the numeric value, compute statistic.
def statisticize(table, index, numeric): 

    statistic = pandas.DataFrame()
    method = [
        'min', 
        'q1', 
        'mean', 
        'median', 
        'q3', 
        'max', 
        'nunique', 
        'sum', 
        'std', 
        'skew', 
        'kurt'
    ]
    for m in method:

        if(m=='min'):     s = table.groupby(index, as_index=False)[numeric].min()
        if(m=='q1'):      s = table.groupby(index, as_index=False)[numeric].quantile(0.25)
        if(m=='mean'):    s = table.groupby(index, as_index=False)[numeric].mean()
        if(m=='median'):  s = table.groupby(index, as_index=False)[numeric].median()
        if(m=='q3'):      s = table.groupby(index, as_index=False)[numeric].quantile(0.75)
        if(m=='max'):     s = table.groupby(index, as_index=False)[numeric].max()
        if(m=='nunique'): s = table.groupby(index, as_index=False)[numeric].nunique()
        if(m=='sum'):     s = table.groupby(index, as_index=False)[numeric].sum()
        if(m=='std'):     s = table.groupby(index, as_index=False)[numeric].std()
        if(m=='skew'):    s = table.groupby(index, as_index=False)[numeric].apply(pandas.DataFrame.skew)
        if(m=='kurt'):    s = table.groupby(index, as_index=False)[numeric].apply(pandas.DataFrame.kurt)        
        c = s.columns.tolist()
        c[1] = '{}_{}'.format(c[1], m)
        s.columns = c
        statistic = pandas.merge(statistic, s, how='outer', on=index) if(not statistic.empty) else s
        continue    

    return(statistic)

##
##
table = sheet.get('history').copy()
index = 'card_id'
variable = [
    'authorized_flag', 'city_id', 'category_1', 'installments', 
    'category_3', 'merchant_category_id', 'month_lag', 
    'purchase_amount', 'category_2', 'state_id', 'subsector_id', 
    'merchant_group_id', 'numerical_1', 'numerical_2', 
    'most_recent_sales_range', 'most_recent_purchases_range', 
    'avg_sales_lag3', 'avg_purchases_lag3', 'active_months_lag3', 
    'avg_sales_lag6', 'avg_purchases_lag6', 'active_months_lag6', 
    'avg_sales_lag12', 'avg_purchases_lag12', 
    'active_months_lag12', 'category_4', 'purchase_date_ymd', 
    'purchase_date_ym', 'purchase_date_y'
]
loop = variable
for i, v in enumerate(tqdm.tqdm(loop)): 
    
    if(i==0): 
        
        group = pandas.DataFrame()
        batch = 0
        pass

    g = statisticize(table=table, index=index, variable=v)
    if(group.empty): group = g
    else: group = pandas.merge(group, g, how='outer', on=index)
    dump = (group.memory_usage().sum() / 1024**3 > 10) if(not group.empty) else False
    if(dump or i==len(loop)-1):
        
        checkpoint = os.path.join(storage, 'statistic_feature_{}.csv'.format(batch))
        group.to_csv(checkpoint, index=False)
        group = pandas.DataFrame()
        batch = batch + 1

    continue

del table

