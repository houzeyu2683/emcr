
##
##  Load packages.
import pandas
import os
import sys

##
##  Record log.
log = './log/overview.txt'
os.makedirs(os.path.dirname(log), exist_ok=True)

##
##  Read sheets.
path = [
    './resource/kaggle/train.csv', 
    './resource/kaggle/test.csv', 
    './resource/kaggle/sample_submission.csv',
    './resource/kaggle/historical_transactions.csv',
    './resource/kaggle/new_merchant_transactions.csv',
    './resource/kaggle/merchants.csv'
]
sheet = [(os.path.basename(p), pandas.read_csv(p, dtype=str)) for p in path]

##
##  The baisc information.
##  Save the information to log folder, 
##  then it can be easy to read. 
sys.stdout = open(log, 'w')
for name, table in sheet:

    print('-'*10, name, '-'*10)
    print(table.head())
    print(table.shape)
    print(table.isna().sum())
    print('-'*80)
    continue

sys.stdout.close()
