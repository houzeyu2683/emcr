
##
##  Load packages.
import pandas
import os
import itertools
import tqdm

##
##  Storage folder.
storage = './resource/kaggle/restructuring/cleaning/engineering/'
os.makedirs(os.path.dirname(storage), exist_ok=True)

##
##  Read sheets.
resource = './resource/kaggle/restructuring/cleaning'
path = [
    os.path.join(resource, 'index.csv'), 
    os.path.join(resource, 'card.csv'), 
    os.path.join(resource, 'history.csv')
]
##  Read the table and set the all value with default type.
sheet = {os.path.basename(p).split(".")[0]: pandas.read_csv(p) for p in path}

##
##  From `index` to `index`.
table = sheet.get('index').copy()
checkpoint = os.path.join(storage, 'index.csv')
key = ['card_id', 'target', 'source']
table[key].to_csv(checkpoint, index=False)

##
##  From `card` to `card`.
table = sheet.get('card')
checkpoint = os.path.join(storage, 'card.csv')
key = [
    'card_id', 'feature_1', 'feature_2', 'feature_3', 
    'first_active_month_label_code', 
    'first_active_year_unit_label_code', 
    'first_active_month_unit_label_code'    
]
table.to_csv(checkpoint, index=False)

##
##  From `history` to `history_combination`.
table = sheet.get('history').copy()

##
##  Combination function.
def combine(table, index, category, numeric, aggregation='sum'):

    pivot = table.pivot_table(values=numeric, index=index, columns=category, aggfunc=aggregation, fill_value=0).copy()
    column = ["{}&{}:{}".format(category, c, numeric) for c in list(pivot.columns)]
    pivot.columns = column
    combination = pivot.reset_index()
    return(combination)

##
##  Assign category and numeric term, ignore some which inappropriate.
index = 'card_id'
_ = [
    "authorized_flag", 'category_1', 'category_3', 'merchant_id',
    'purchase_date', 'merchant_group_id', 'most_recent_sales_range', 
    'most_recent_purchases_range', 'category_4', 
    'purchase_date_year_month_day_unit', 'purchase_date_year_month_unit',
    "purchase_date_year_unit", "merchant_id_label_code", 
    "purchase_date_label_code"
]
category = [
    'installments', 'month_lag', 'category_2', 'state_id', 
    'subsector_id', 'active_months_lag3', 'active_months_lag6', 
    'active_months_lag12', 'authorized_flag_label_code', 
    "category_1_label_code", "category_3_label_code", 
    "most_recent_sales_range_label_code", 
    "most_recent_purchases_range_label_code", "category_4_label_code", 
    "purchase_date_year_month_unit_label_code", 
    "purchase_date_year_unit_label_code" 
]
numeric = [
    'city_id', 'merchant_category_id', 'purchase_amount', 'numerical_1',
    'numerical_2', 'avg_sales_lag3', 'avg_purchases_lag3', 
    'avg_sales_lag6', 'avg_purchases_lag6', 'avg_sales_lag12', 
    'avg_purchases_lag12', 'purchase_date_year_month_day_unit_label_code'
]
loop = list(itertools.product(category, numeric))
for i, (c, n) in enumerate(tqdm.tqdm(loop)): 

    if(i==0): group, batch = pandas.DataFrame(), 0
    g = combine(table=table, index=index, category=c, numeric=n, aggregation='sum')
    # if(group.empty): group = g
    # else: group = pandas.merge(group, g, how='outer', on=index)
    group = g if(group.empty) else pandas.merge(group, g, how='outer', on=index)
    pass

    dump = (group.memory_usage().sum() / 1024**3 > 1) if(not group.empty) else False
    last = i==len(loop)-1
    if(dump or last):
        
        checkpoint = os.path.join(storage, 'history_combination_{}.csv'.format(batch))
        group.to_csv(checkpoint, index=False)
        group = pandas.DataFrame()
        batch = batch + 1

    continue

##
##  From `history` to `history_statistic`.
table = sheet.get('history').copy()

##
##  Statistic function.
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
##  Assign numeric term, ignore some which inappropriate.
index = 'card_id'
numeric = [
    "city_id", "merchant_category_id", "purchase_amount",
    "merchant_group_id", "numerical_1", "numerical_2",
    "avg_sales_lag3", "avg_purchases_lag3", "avg_sales_lag6",
    "avg_purchases_lag6", "avg_sales_lag12", "avg_purchases_lag12", 
    "merchant_id_label_code", "purchase_date_label_code", 
    "purchase_date_year_month_day_unit_label_code"
]
loop = numeric
for i, n in enumerate(tqdm.tqdm(loop)): 
    
    if(i==0): 
        
        group = pandas.DataFrame()
        batch = 0
        pass

    g = statisticize(table=table, index=index, numeric=n)
    if(group.empty): group = g
    else: group = pandas.merge(group, g, how='outer', on=index)
    dump = (group.memory_usage().sum() / 1024**3 > 5) if(not group.empty) else False
    if(dump or i==len(loop)-1):
        
        checkpoint = os.path.join(storage, 'history_statistic_{}.csv'.format(batch))
        group.to_csv(checkpoint, index=False)
        group = pandas.DataFrame()
        batch = batch + 1

    continue

