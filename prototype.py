
##
##  Load packages.
import pandas
import os

##
##  Storage folder.
storage = './resource/prototype/'
os.makedirs(os.path.dirname(storage), exist_ok=True)

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
##  Combine the train and test together.
table = {'train': sheet[0][1], 'test': sheet[1][1]}
table['test']['target'] = None
table['index'] = pandas.concat([table['train'], table['test']], axis=0)
table['index'].to_csv(os.path.join(storage, 'index.csv'), index=False)
del table

##
##  Combine the historical transaction and merchant together.
table = {
    'transaction-0': sheet[3][1], 
    'transaction-1': sheet[4][1], 
    'merchant':sheet[5][1]
}
##  Two transaction table merge and remove repeat row if need.
table['transaction'] = pandas.concat([table['transaction-0'], table['transaction-1']], axis=0)
table['transaction'] = table['transaction'].drop_duplicates()
##  The merchant table and transaction table have some same column,
##  but the `merchant_id` key for merge.
##  Then remove the repeat column in the merchant table.
key = 'merchant_id'
repeat = list(set(table['transaction'].keys()) & set(table['merchant'].keys()))
repeat.remove(key)
table['merchant'] = table['merchant'].drop(repeat, axis=1)
##  And the merchant table have some problem,
##  the 'merchant_id' is not unique, 
##  then drop and keep the last in the table.
table['merchant'] = table['merchant'].drop_duplicates(subset=[key], keep='last')
##  Merge the transaction table and merchant table together, join with transaction table.
##  So if some 'merchant_id' exist in the transaction table but not in the merchant table,
##  that mean value of column from the merchant table is missing value, we will need to handle it.
##  We call the big table denote by the history table.
table['history'] = pandas.merge(table['transaction'], table['merchant'], how='left', on=key)
table['history'].to_csv(os.path.join(storage, 'history.csv'), index=False)

