
##
##  Load packages.
import pandas
import os

##
##  Read sheets.
resource = './resource/kaggle/'
path = [
    os.path.join(resource, 'train.csv'), 
    os.path.join(resource, 'test.csv'), 
    os.path.join(resource, 'sample_submission.csv'),
    os.path.join(resource, 'historical_transactions.csv'),
    os.path.join(resource, 'new_merchant_transactions.csv'),
    os.path.join(resource, 'merchants.csv')
]
sheet = {os.path.basename(p).split('.')[0]: pandas.read_csv(p, dtype=str) for p in path}
sheet.keys()

##
##  Storage folder.
storage = './resource/kaggle/prototype/'
os.makedirs(os.path.dirname(storage), exist_ok=True)

##
##  Combine `train` and `test` to `card`.
table = {'train': sheet.get('train'), 'test': sheet.get('test')}
table['train']['mode'] = 0
table['test']['mode'] = 1
table['test']['target'] = None
table['card'] = pandas.concat([table['train'], table['test']], axis=0)
table['card'].to_csv(os.path.join(storage, 'card.csv'), index=False)
del table

##
##  Combine `historical_transactions`, `new_merchant_transactions` and `merchant`.
table = {
    'historical_transactions': sheet.get("historical_transactions"), 
    'new_merchant_transactions': sheet.get("new_merchant_transactions"), 
    'merchant':sheet.get("merchants")
}
##  Merge `historical_transactions` and `new_merchant_transactions` to `transaction`.
##  Then remove repeat row.
table['transaction'] = pandas.concat([table['historical_transactions'], table['new_merchant_transactions']], axis=0)
table['transaction'] = table['transaction'].drop_duplicates()
##  Find `merchant` and `transaction` have some same columns,
##  Remove some columns of `merchant` if `transaction` already exist.
##  Keep `merchant_id` exist and merge later.
key = 'merchant_id'
repeat = list(set(table['transaction'].keys()) & set(table['merchant'].keys()))
repeat.remove(key)
table['merchant'] = table['merchant'].drop(repeat, axis=1)
##  Found 'merchant_id' of `merchant` is not unique,
##  then drop and keep the last in the table.
table['merchant'] = table['merchant'].drop_duplicates(subset=[key], keep='last')
##  Insert `merchant` to `transaction`, denote `history`.
table['history'] = pandas.merge(table['transaction'], table['merchant'], how='left', on=key)
table['history'].to_csv(os.path.join(storage, 'history.csv'), index=False)
del table
