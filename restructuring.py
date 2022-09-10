
##
##  Load packages.
import pandas
import os

##
##  Storage folder.
storage = './resource/kaggle/restructuring/'
os.makedirs(os.path.dirname(storage), exist_ok=True)

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
##  Read the table and set the all value with string type.
sheet = {os.path.basename(p).split(".")[0]: pandas.read_csv(p, dtype=str) for p in path}

##
##  Combine `train` and `test` to `group`.
##  In `test`, fill special symbol to `target` column.
table = {'train': sheet.get('train'), 'test': sheet.get('test')}
table['test']['target'] = -10000
table['train']['source'] = 'train'
table['test']['source'] = 'test'
table['group'] = pandas.concat([table['train'], table['test']], axis=0)
##  Save to `index`.
key = ['card_id', 'target', 'source']
table['group'][key].to_csv(os.path.join(storage, 'index.csv'), index=False)
##  Save to `card`.
key = ['card_id', 'first_active_month', 'feature_1', 'feature_2', 'feature_3']
table['group'][key].to_csv(os.path.join(storage, 'card.csv'), index=False)

##
##  Combine `historical_transaction` and `new_merchant_transactions` to `transaction`.
table = {
    'historical_transactions': sheet.get("historical_transactions"), 
    'new_merchant_transactions': sheet.get("new_merchant_transactions"), 
    'merchant':sheet.get('merchants')
}
table['transaction'] = pandas.concat(
    [table['historical_transactions'], table['new_merchant_transactions']], 
    axis=0
)
table['transaction'] = table['transaction'].drop_duplicates()
##  Found some columns both in `merchant` and `transaction`.
##  remove the duplicated columns in the `merchant` but not `merchant_id`.
key = 'merchant_id'
repeat = list(set(table['transaction'].keys()) & set(table['merchant'].keys()))
repeat.remove(key)
table['merchant'] = table['merchant'].drop(repeat, axis=1)
##  The 'merchant_id' of `merchant` not unique, then drop and keep the last.
table['merchant'] = table['merchant'].drop_duplicates(subset=[key], keep='last')
##  Merge `merchant` to `transaction`, denote by 'history'.
##  Some `merchant_id` in `transaction` but not in the `merchant`,
##  then they will be missing value.
table['history'] = pandas.merge(table['transaction'], table['merchant'], how='left', on=key)
table['history'].to_csv(os.path.join(storage, 'history.csv'), index=False)

