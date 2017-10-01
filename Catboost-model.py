..

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


train.head()
Out[17]:
ID	datetime	siteid	offerid	category	merchant	countrycode	browserid	devid	click
0	IDsrk7SoW	2017-01-14 09:42:09	4709696.0	887235	17714	20301556	e	Firefox	NaN	0
1	IDmMSxHur	2017-01-18 17:50:53	5189467.0	178235	21407	9434818	b	Mozilla Firefox	Desktop	0
2	IDVLNN0Ut	2017-01-11 12:46:49	98480.0	518539	25085	2050923	a	Edge	NaN	0
3	ID32T6wwQ	2017-01-17 10:18:43	8896401.0	390352	40339	72089744	c	Firefox	Mobile	0
4	IDqUShzMg	2017-01-14 16:02:33	5635120.0	472937	12052	39507200	d	Mozilla Firefox	Desktop	0
In [18]:

test.head()
Out[18]:
ID	datetime	siteid	offerid	category	merchant	countrycode	browserid	devid
0	IDFDJVI	2017-01-22 09:55:48	755610.0	808980	17714	26391770	b	Mozilla Firefox	Desktop
1	IDNWkTQ	2017-01-22 03:54:39	3714899.0	280355	12052	39507200	b	Edge	Tablet
2	ID9pRmM	2017-01-21 10:25:50	4378333.0	930819	30580	46148550	e	Mozilla Firefox	NaN
3	IDHaQaj	2017-01-22 14:45:53	1754730.0	612234	11837	8837581	b	Edge	Tablet
4	IDT2CrF	2017-01-22 09:34:07	5299909.0	524289	45620	31388981	b	Mozilla	NaN
In [3]:
# check missing values per column
train.isnull().sum(axis=0)/train.shape[0]
Out[3]:
ID             0.000000
datetime       0.000000
siteid         0.099896
offerid        0.000000
category       0.000000
merchant       0.000000
countrycode    0.000000
browserid      0.050118
devid          0.149969
click          0.000000
dtype: float64



train['siteid'].fillna(-9999, inplace=True)
test['siteid'].fillna(-9999, inplace=True)

train['browserid'].fillna("None", inplace=True)
test['browserid'].fillna("None", inplace=True)

train['devid'].fillna("None", inplace=True)
test['devid'].fillna("None", inplace=True)


train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])
In [6]:
# create datetime variable
train['tweekday'] = train['datetime'].dt.weekday
train['thour'] = train['datetime'].dt.hour
train['tminute'] = train['datetime'].dt.minute

test['tweekday'] = test['datetime'].dt.weekday
test['thour'] = test['datetime'].dt.hour
test['tminute'] = test['datetime'].dt.minute
In [7]:
cols = ['siteid','offerid','category','merchant']

for x in cols:
    train[x] = train[x].astype('object')
    test[x] = test[x].astype('object')
In [8]:
cols_to_use = list(set(train.columns) - set(['ID','datetime','click']))


cat_cols = [0,1,2,4,6,7,8]


# modeling on sampled (1e6) rows
rows = np.random.choice(train.index.values, 1e6)
sampled_train = train.loc[rows]
/home//anaconda2/envs/py35/lib/python3.5/site-packages/ipykernel/__main__.py:1: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future
  if __name__ == '__main__':
In [15]:
trainX = sampled_train[cols_to_use]
trainY = sampled_train['click']


X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size = 0.30)
model = CatBoostClassifier(depth=10, iterations=10, learning_rate=0.1, eval_metric='AUC', random_seed=1)

model.fit(X_train
          ,y_train
          ,cat_features=cat_cols
          ,eval_set = (X_test, y_test)
          ,use_best_model = True
         )


pred = model.predict_proba(test[cols_to_use])[:,1]
In [28]:
sub = pd.DataFrame({'ID':test['ID'],'click':pred})
sub.to_csv('cb_sub1.csv',index=False)
