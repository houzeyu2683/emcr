
##
##  針對執行思路進行簡短描述，首先確認每個變數欄位的型態，
##  對於數字狀態，無論是類別或是數值，建議都平移到正實數空間，
##  如此一來，數字狀態的遺失值填補可以使用負數進行填補，文字狀態則可以填補字段。
##  所以接著逐一的檢查欄位，適當地填補遺失值或極端值。
##  針對文字狀態的類別變數進行適當地顆粒度細分或粗和，產生延伸變數。
##  將文字狀態的類別變數進行適當地編碼，用新的欄位來儲存，
##  藉此做到保留原有欄位，增加後續調整的彈性。

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

##
##  Get `card` table.
table = sheet.get('card').copy()
##  Process individually.
##  Handle `first_active_month` column.
name = 'first_active_month'
column = table[name].copy()
column = column.astype("str")
column[column=='nan'] = "2017-09"
table[name] = column
##  Handle 'card_id' column.
name = 'card_id'
column = table[name].copy()
column = column.astype("str")
table[name] = column
##  Handle 'feature_1' column.
name = 'feature_1'
column = table[name].copy()
column = column.astype("int64")
column = column - column.min()
table[name] = column
##  Handle 'feature_2' column.
name = 'feature_2'
column = table[name].copy()
column = column.astype('int64')
table[name] = column
column = column - column.min()
##  Handle 'feature_3' column.
name = 'feature_3'
column = table[name].copy()
column = column.astype('int64')
column = column - column.min()
table[name] = column
##  Handle 'target' column.
name = 'target'
column = table[name].copy()
column = column.astype('float64')
table[name] = column
##  Handle 'source' column.
name = 'source'
column = table[name].copy()
column = column.astype("str")
table[name] = column

##
##  Extension of column.
name = 'first_active_month'
column = table[name].copy()
table['extension_first_active_month_partial_year'] = [i.split('-')[0] for i in column]
table['extension_first_active_month_partial_month'] = [i.split('-')[1] for i in column]

##
##  Character category to code.
loop = [
    'first_active_month', 
    'extension_first_active_month_partial_year', 
    'extension_first_active_month_partial_month'
]
tag = 'label_code'
for c in loop:

    encoder = sklearn.preprocessing.LabelEncoder()
    _ = encoder.fit(table[c])
    table["{}_{}".format(c, tag)] = encoder.transform(table[c])
    continue

table.head()
key = ['card_id', 'target', 'source']
table[key].to_csv(os.path.join(storage, 'index.csv'), index=False)
key = [
    'card_id', 'first_active_month', 'feature_1', 'feature_2', 'feature_3',
    'extension_first_active_month_partial_year', 'extension_first_active_month_partial_year', 
    'extension_first_active_month_partial_year_label_code',
    'extension_first_active_month_partial_year_label_code'
]
table[key].to_csv(os.path.join(storage, 'card.csv'), index=False)

##
##  Get `history` table.
table = sheet.get('history').copy()
##  Process individually.
##  Handle `authorized_flag` column.
name = 'authorized_flag'
column = table[name].copy()
column = column.astype("str")
table[name] = column
##  Handle `card_id` column.
name = 'card_id'
column = table[name].copy()
column = column.astype("str")
table[name] = column
##  Handle 'city_id' column.
name = 'city_id'
column = table[name].copy()
column = column.astype("int64")
column = column - column.min() 
table[name] = column
##  Handle 'category_1' column.
name = 'category_1'
column = table[name].copy()
column = column.astype("str")
table[name] = column
##  Handle 'installments' column.
name = 'installments'
column = table[name].copy()
column = column.astype("int64")
column = column - column.min()
table[name] = column
##  Handle 'category_3' column.
name = 'category_3'
column = table[name].copy()
column = column.astype("str")
column[column=='nan'] = "-1"
table[name] = column
##  Handle 'merchant_category_id' column.
name = 'merchant_category_id'
column = table[name].copy()
column = column.astype("int")
column = column - column.min()
table[name] = column
##  Handle 'merchant_id' column.
name = 'merchant_id'
column = table[name].copy()
column = column.astype("str")
column[column=='nan'] = "-1"
table[name] = column
##  Handle 'month_lag' column.
name = 'month_lag'
column = table[name].copy()
column = column.astype("int64")
column = column - column.min()
table[name] = column
##  Handle 'purchase_amount' column.
name = 'purchase_amount'
column = table[name].copy()
column = column.astype("float64")
column = column - column.min()
table[name] = column
##  Handle 'purchase_date' column.
name = 'purchase_date'
column = table[name].copy()
column = column.astype("str")
table[name] = column
##  Handle 'category_2' column.
name = 'category_2'
column = table[name].copy()
column = column.astype("float64")
column = column - column.min()
column = column.fillna(-1)
table[name] = column
##  Handle 'state_id' column.
name = 'state_id'
column = table[name].copy()
column = column.astype("int64")
column = column - column.min()
table[name] = column
##  Handle 'subsector_id' column.
name = 'subsector_id'
column = table[name].copy()
column = column.astype("int64")
column = column - column.min()
table[name] = column
##  Handle 'merchant_group_id' column.
name = 'merchant_group_id'
column = table[name].copy()
column = column.astype('Int64')
column = column - column.min()
column = column.fillna(-1)
table[name] = column
##  Handle 'numerical_1' column.
name = 'numerical_1'
column = table[name].copy()
column = column.astype('float64')
column = column - column.min()
column = column.fillna(-1)
table[name] = column
##  Handle 'numerical_2' column.
name = 'numerical_2'
column = table[name].copy()
column = column.astype('float64')
column = column - column.min()
column = column.fillna(-1)
table[name] = column
##  Handle 'most_recent_sales_range' column.
name = 'most_recent_sales_range'
column = table[name].copy()
column = column.astype('str')
column[column=='nan'] = "-1"
table[name] = column
##  Handle 'most_recent_purchases_range' column.
name = 'most_recent_purchases_range'
column = table[name].copy()
column = column.astype('str')
column[column=='nan'] = "-1"
table[name] = column
##  Handle 'avg_sales_lag3' column.
name = 'avg_sales_lag3'
column = table[name].copy()
column = column.astype('float64')
column = column - column.min()
column = column.fillna(-1)
table[name] = column
##  Handle 'avg_purchases_lag3' column.
name = 'avg_purchases_lag3'
column = table[name].copy()
column = column.astype('float64')
column = column - column.min()
ceiling = column[column!=numpy.inf].max() + column[column!=numpy.inf].std()
column[column==numpy.inf] = ceiling
column = column.fillna(-1)
table[name] = column
##  Handle 'active_months_lag3' column.
name = 'active_months_lag3'
column = table[name].copy()
column = column.astype('Int64')
column = column - column.min()
column = column.fillna(-1)
table[name] = column
##  Handle 'avg_sales_lag6' column.
name = 'avg_sales_lag6'
column = table[name].copy()
column = column.astype('float64')
column = column - column.min()
column = column.fillna(-1)
table[name] = column
##  Handle 'avg_purchases_lag6' column.
name = 'avg_purchases_lag6'
column = table[name].copy()
column = column.astype('float64')
column = column - column.min()
ceiling = column[column!=numpy.inf].max() + column[column!=numpy.inf].std()
column[column==numpy.inf] = ceiling
column = column.fillna(-1)
table[name] = column
##  Handle 'active_months_lag6' column.
name = 'active_months_lag6'
column = table[name].copy()
column = column.astype('Int64')
column = column - column.min()
column = column.fillna(-1)
table[name] = column
##  Handle 'avg_sales_lag12' column.
name = 'avg_sales_lag12'
column = table[name].copy()
column = column.astype('float64')
column = column - column.min()
column = column.fillna(-1)
table[name] = column
##  Handle 'avg_purchases_lag12' column.
name = 'avg_purchases_lag12'
column = table[name].copy()
column = column.astype('float64')
column = column - column.min()
ceiling = column[column!=numpy.inf].max() + column[column!=numpy.inf].std()
column[column==numpy.inf] = ceiling
column = column.fillna(-1)
table[name] = column
##  Handle 'active_months_lag12' column.
name = 'active_months_lag12'
column = table[name].copy()
column = column.astype('Int64')
column = column - column.min()
column = column.fillna(-1)
table[name] = column
##  Handle 'category_4' column.
name = 'category_4'
column = table[name].copy()
column = column.astype('str')
column[column=='nan'] = "-1"
table[name] = column

##
##  Extension of column.
origin = 'purchase_date'
series = table[origin].copy()
table['purchase_date(ymd)'] = [i.split(" ")[0] for i in series]
table['purchase_date(ym)' ] = ["-".join(i.split(" ")[0].split('-')[0:2]) for i in series]
table['purchase_date(y)'  ] = ["-".join(i.split(" ")[0].split('-')[0:1]) for i in series]

##
##  Character category to code.
loop = [
    'authorized_flag', 'category_1', 'category_3', 'merchant_id',
    'most_recent_sales_range', 'most_recent_purchases_range', 'category_4',
    'purchase_date(ymd)', 'purchase_date(ym)', 'purchase_date(y)'
]
tag = 'code'
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
