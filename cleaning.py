
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
    os.path.join(resource, 'index.csv'),
    os.path.join(resource, 'card.csv'), 
    os.path.join(resource, 'history.csv')
]
##  Read the table and set the all value with string type.
sheet = {os.path.basename(p).split(".")[0]: pandas.read_csv(p, dtype=str) for p in path}

##
##  Get `index` table.
table = sheet.get('index').copy()
##  Process individually.
##  Handle 'card_id' column.
name = 'card_id'
column = table[name].copy()
column = column.astype("str")
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
checkpoint = os.path.join(storage, 'index.csv')
# key = ['card_id', 'target', 'source']
# table[key].to_csv(checkpoint, index=False)
table.to_csv(checkpoint, index=False)

##
##  Get `card` table.
table = sheet.get('card').copy()
##  Process individually.
##  Handle 'card_id' column.
name = 'card_id'
column = table[name].copy()
column = column.astype("str")
table[name] = column
##  Handle `first_active_month` column.
name = 'first_active_month'
column = table[name].copy()
column = column.astype("str")
column[column=='nan'] = "2017-09"
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

##
##  Extension of column.
name = 'first_active_month'
column = table[name].copy()
table['first_active_year_unit'] = [i.split('-')[0] for i in column]
table['first_active_month_unit'] = [i.split('-')[1] for i in column]

##
##  Character category to code.
loop = [
    'first_active_month', 
    'first_active_year_unit', 
    'first_active_month_unit'
]
tag = 'label_code'
for c in loop:

    encoder = sklearn.preprocessing.LabelEncoder()
    _ = encoder.fit(table[c])
    table["{}_{}".format(c, tag)] = encoder.transform(table[c])
    continue

checkpoint = os.path.join(storage, 'card.csv')
# key = [
#     'card_id',
#     "feature_1", 
#     "feature_2",
#     "feature_3",
#     "first_active_month_label_code",
#     "first_active_year_unit_label_code",
#     "first_active_month_unit_label_code"
# ]
# table[key].to_csv(checkpoint, index=False)
table.to_csv(checkpoint, index=False)

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
name = 'purchase_date'
column = table[name].copy()
table['purchase_date_year_month_day_unit'] = [i.split(" ")[0] for i in column]
table['purchase_date_year_month_unit'] = ["-".join(i.split(" ")[0].split('-')[0:2]) for i in column]
table['purchase_date_year_unit'] = ["-".join(i.split(" ")[0].split('-')[0:1]) for i in column]

##
##  Character category to code.
loop = [
    'authorized_flag', 'category_1', 'category_3', 'merchant_id', 'purchase_date',
    'most_recent_sales_range', 'most_recent_purchases_range', 'category_4',
    'purchase_date_year_month_day_unit', 'purchase_date_year_month_unit', 
    'purchase_date_year_unit'
]
tag = 'label_code'
for c in loop:

    encoder = sklearn.preprocessing.LabelEncoder()
    _ = encoder.fit(table[c])
    table["{}_{}".format(c, tag)] = encoder.transform(table[c])
    continue

checkpoint = os.path.join(storage, 'history.csv')
# key = [
#     "card_id",
#     "city_id",
#     'installments',
#     'merchant_category_id',
#     'month_lag',
#     'purchase_amount',
#     'category_2',
#     'state_id',
#     'subsector_id',
#     'merchant_group_id',
#     'numerical_1',
#     'numerical_2',
#     'avg_sales_lag3',
#     'avg_purchases_lag3',
#     'active_months_lag3',
#     'avg_sales_lag6',
#     'avg_purchases_lag6',
#     'active_months_lag6',
#     'avg_sales_lag12',
#     'avg_purchases_lag12',
#     'active_months_lag12',
#     'authorized_flag_label_code', 
#     'category_1_label_code', 
#     'category_3_label_code', 
#     'merchant_id_label_code', 
#     'purchase_date_label_code', 
#     'most_recent_sales_range_label_code', 
#     'most_recent_purchases_range_label_code', 
#     'category_4_label_code', 
#     'purchase_date_year_month_day_unit_label_code', 
#     'purchase_date_year_month_unit_label_code', 
#     'purchase_date_year_unit_label_code', 
# ]
# table[key].to_csv(checkpoint, index=False)
table.to_csv(checkpoint, index=False)