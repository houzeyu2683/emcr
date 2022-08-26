
##
##  Load packages.
import pandas
import os

##
##  Storage folder.
storage = './resource/kaggle/prototype/'
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
##  Read the table and set the all value with string type.
sheet = {os.path.basename(p).split(".")[0]: pandas.read_csv(p, dtype=str) for p in path}

##
##  Combine `train` and `test` together, save to `card`.
table = {'train': sheet.get('train'), 'test': sheet.get('test')}
table['test']['target'] = -10000
table['train']['source'] = 'train'
table['test']['source'] = 'test'
table['card'] = pandas.concat([table['train'], table['test']], axis=0)
table['card'].to_csv(os.path.join(storage, 'card.csv'), index=False)
del table

##
##  Connect `historical_transaction` and `new_merchant_transactions` to one file,
##  then merge `merchant` together.
table = {
    'historical_transactions': sheet.get("historical_transactions"), 
    'new_merchant_transactions': sheet.get("new_merchant_transactions"), 
    'merchant':sheet.get('merchants')
}
##  Merge `historical_transaction` and `new_merchant_transactions` to `transaction`,
##  remove repeat row if need.
table['transaction'] = pandas.concat(
    [table['historical_transactions'], table['new_merchant_transactions']], 
    axis=0
)
table['transaction'] = table['transaction'].drop_duplicates()
##  Found some columns both in `merchant` and `transaction`.
##  However keep `merchant_id` key,
##  then remove the other columns of repeat in the `merchant`.
key = 'merchant_id'
repeat = list(set(table['transaction'].keys()) & set(table['merchant'].keys()))
repeat.remove(key)
table['merchant'] = table['merchant'].drop(repeat, axis=1)
##  Found 'merchant_id' of `merchant` not unique,
##  then drop and keep the last.
table['merchant'] = table['merchant'].drop_duplicates(subset=[key], keep='last')
##  Merge `merchant` to `transaction`, denote by 'history'.
##  The fact is some `merchant_id` in `transaction` and not in the `merchant`,
##  then the `merchant` information will be missing value.
table['history'] = pandas.merge(table['transaction'], table['merchant'], how='left', on=key)
table['history'].to_csv(os.path.join(storage, 'history.csv'), index=False)

