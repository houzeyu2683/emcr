
##
##  Load packages.
import pandas
import numpy
import os

##
##  Read sheets.
resource = "./resource/kaggle/prototype/"
path = [
    os.path.join(resource, 'card.csv'), 
    os.path.join(resource, 'history.csv')
]
sheet = {os.path.basename(p).split(".")[0]: pandas.read_csv(p, dtype=str) for p in path}

##
##  Storage folder.
storage = './resource/kaggle/prototype/clean/'
os.makedirs(os.path.dirname(storage), exist_ok=True)

##
##  Handle `card`.
key = 'card'
table = sheet.get(key)
table.head()




table['first_active_month'] = table['first_active_month'].astype(str)

table['first_active_month'].isna().sum()
table['card_id'] = table['card_id'].astype(str)
table['feature_1'] = table['feature_1'].astype(int)


table['feature_1'] = table['feature_1'].astype(float)
table['feature_2'] = table['feature_2'].astype(float)
table['feature_3'] = table['feature_3'].astype(float)
table['target']    = table['target'].astype(float)
table['mode']      = table['mode'].astype(float)
table.loc[table['first_active_month'].isna()] = '2017-09'
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
##  Before fill missing value of number, limit the inf to a higher value.
table.loc[table['avg_purchases_lag3']==numpy.inf, 'avg_purchases_lag3'] = 10000 + 61851.33333333
table.loc[table['avg_purchases_lag6']==numpy.inf, 'avg_purchases_lag6'] = 10000 + 56077.5
table.loc[table['avg_purchases_lag12']==numpy.inf, 'avg_purchases_lag12'] = 10000 + 50215.55555556
for i in table:

    if(table[i].dtype == float): table[i] = table[i] - table[i].min()
    continue

for i in table:

    if(table[i].dtype == float): table[i] = table[i].fillna(-1.0)
    if(table[i].dtype != float): table[i] = table[i].fillna('<NA>')
    continue

table.to_csv(os.path.join(storage, 'history.csv'), index=False)
del table

