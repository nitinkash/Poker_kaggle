__author__ = 'nitinkashyap'
from sklearn.svm import SVR
import pandas as pd

reader = pd.read_csv('train.csv')
out = reader['hand'].tolist()
print('Here')
del reader['hand']
values = reader.values
test = pd.read_csv('test.csv')
ids = test['id']
del test['id']
te_val = test.values
k = SVR()
X = k.fit(values, out)
y = k.predict(te_val)
final = pd.DataFrame()
final['id'] = ids
final['hand'] = y
final.to_csv('Submission1.csv')



