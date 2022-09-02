
##
##  Load packages.
import pandas
import os
import functools

##
##  Storage folder.
storage = './resource/kaggle/restructuring/cleaning/engineering/preparing'
os.makedirs(os.path.dirname(storage), exist_ok=True)

##
##  Read sheets.
resource = './resource/kaggle/restructuring/cleaning/engineering'
path = [
    os.path.join(resource, 'index.csv'), 
    os.path.join(resource, 'card.csv'), 
    os.path.join(resource, 'history_combination_feature_0.csv'),
    os.path.join(resource, 'history_combination_feature_1.csv'),
    os.path.join(resource, 'history_combination_feature_2.csv'),
    os.path.join(resource, 'history_combination_feature_3.csv'),
    os.path.join(resource, 'history_combination_feature_4.csv'),
    os.path.join(resource, 'history_combination_feature_5.csv'),
    os.path.join(resource, 'history_statistic_feature_0.csv')
]
##  Read the table and set the all value with default type.
sheet = {os.path.basename(p).split(".")[0]: pandas.read_csv(p) for p in path}

##
##
index = 'card_id'
gather = lambda x, y: pandas.merge(x, y, how='outer', on=index) 
integration = functools.reduce(gather, sheet.values())

integration.loc[integration['source']=='train'].shape
