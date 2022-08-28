
##
##  Load packages.
import pandas
import numpy
import os
import sklearn.preprocessing

##
##  Read sheets.
resource = "./resource/kaggle/prototype/clean/"
path = [
    os.path.join(resource, 'card.csv'), 
    os.path.join(resource, 'history.csv')
]
sheet = {os.path.basename(p).split(".")[0]: pandas.read_csv(p, dtype=str) for p in path}

##
##  Storage folder.
storage = './resource/kaggle/prototype/clean/scatter/'
os.makedirs(os.path.dirname(storage), exist_ok=True)

##
##  Handle `card`.
key = 'card'
table = sheet.get(key)
table['feature_1'] = table['feature_1'].astype(float)
table['feature_2'] = table['feature_2'].astype(float)
table['feature_3'] = table['feature_3'].astype(float)
table['target']    = table['target'].astype(float)
table['mode']      = table['mode'].astype(float)
##  Generate variable.
table['first_active_year_unit']  = [i.split('-')[0] for i in table['first_active_month']]
table['first_active_month_unit'] = [i.split('-')[1] for i in table['first_active_month']]
##  Convert to code.
column = ['first_active_month', 'first_active_year_unit', 'first_active_month_unit']
for c in column:
    
    engine = sklearn.preprocessing.LabelEncoder()
    _ = engine.fit(table[c])
    table[c] = engine.transform(table[c])
    continue

table.to_csv(os.path.join(storage, 'card.csv'), index=False)
del table

##
##  Handle missing value of `history`.
key = 'history'
table = sheet.get(key)
table['city_id']              = table['city_id'].astype(float)
table['installments']         = table['installments'].astype(float)
table['merchant_category_id'] = table['merchant_category_id'].astype(float)
table['month_lag']            = table['month_lag'].astype(float)
table['purchase_amount']      = table['purchase_amount'].astype(float)
table['category_2']           = table['category_2'].astype(float)
table['state_id']             = table['state_id'].astype(float)
table['subsector_id']         = table['subsector_id'].astype(float)
table['merchant_group_id']    = table['merchant_group_id'].astype(float)
table['numerical_1']          = table['numerical_1'].astype(float)
table['numerical_2']          = table['numerical_2'].astype(float)
table['avg_sales_lag3']       = table['avg_sales_lag3'].astype(float)
table['avg_purchases_lag3']   = table['avg_purchases_lag3'].astype(float)
table['active_months_lag3']   = table['active_months_lag3'].astype(float)
table['avg_sales_lag6']       = table['avg_sales_lag6'].astype(float)
table['avg_purchases_lag6']   = table['avg_purchases_lag6'].astype(float)
table['active_months_lag6']   = table['active_months_lag6'].astype(float)
table['avg_sales_lag12']      = table['avg_sales_lag12'].astype(float)
table['avg_purchases_lag12']  = table['avg_purchases_lag12'].astype(float)
table['active_months_lag12']  = table['active_months_lag12'].astype(float)
##  Generate variable.
table['purchase_date_month'] = [i.split(' ')[0] for i in table['purchase_date']]
##  Convert to code.
column = [
    'authorized_flag', 'category_1', 'category_3', 'merchant_id',
    'most_recent_sales_range', 'most_recent_purchases_range', 'category_4'
]
for c in column:
    
    engine = sklearn.preprocessing.LabelEncoder()
    _ = engine.fit(table[c])
    table[c] = engine.transform(table[c])
    continue

table.to_csv(os.path.join(storage, 'history.csv'), index=False)
del table



