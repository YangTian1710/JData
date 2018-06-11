# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import metrics
import sklearn.preprocessing as sp
import matplotlib.pyplot as plt
import seaborn as sns

# In[ ]:


train = pd.read_csv('../data/train_featureVB.csv')
test = pd.read_csv('../data/test_featureVB.csv')

X_train = train.drop(['uid','label'],axis=1)
X_test = test.drop(['uid'],axis=1)

# In[ ]:


dtrain = lgb.Dataset(X_train,label=train.label)
dtest = lgb.Dataset(X_test)

# In[ ]:


def evalMetric(preds, dtrain):
    label = dtrain.get_label()

    pre = pd.DataFrame({'preds': preds, 'label': label})
    pre = pre.sort_values(by='preds', ascending=False)

    auc = metrics.roc_auc_score(pre.label, pre.preds)

    pre.preds = pre.preds.map(lambda x: 1 if x >= 0.5 else 0)

    f1 = metrics.f1_score(pre.label, pre.preds)

    res = 0.6 * auc + 0.4 * f1

    return 'res', res, True


# In[ ]:

### 设置初始参数--不含交叉验证参数
print('设置参数')
params = {
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric': 'binary_logloss',
          }

### 交叉验证(调参)
print('交叉验证')
min_merror = float('Inf')
best_params = {}

# 准确率
print("调参1：提高准确率")
for num_leaves in range(20,200,5):
    for max_depth in range(3,8,1):
        params['num_leaves'] = num_leaves
        params['max_depth'] = max_depth

        cv_results = lgb.cv(
                            params,
                            dtrain,
                            feval=evalMetric,
                            seed=2018,
                            nfold=3,
                            metrics=['binary_error'],
                            early_stopping_rounds=10,
                            verbose_eval=True
                            )

        mean_merror = pd.Series(cv_results['binary_error-mean']).min()
        boost_rounds = pd.Series(cv_results['binary_error-mean']).argmin()

        if mean_merror < min_merror:
            min_merror = mean_merror
            best_params['num_leaves'] = num_leaves
            best_params['max_depth'] = max_depth

params['num_leaves'] = best_params['num_leaves']
params['max_depth'] = best_params['max_depth']

# ### 本地CV

# In[ ]:


lgb.cv(params, dtrain, feval=evalMetric, early_stopping_rounds=100, verbose_eval=5, num_boost_round=10000, nfold=3,
       metrics=['evalMetric'])

# ## 训练

# In[ ]:


model = lgb.train(params, dtrain,
                  feval=evalMetric,
                  verbose_eval=5, num_boost_round=300, valid_sets=[dtrain])

# ### 预测

# In[ ]:


pred = model.predict(test.drop(['uid'], axis=1))

# In[ ]:


res = pd.DataFrame({'uid': test.uid, 'label': pred})

# In[ ]:


res = res.sort_values(by='label', ascending=False)
res.label = res.label.map(lambda x: 1 if x >= 0.5 else 0)
res.label = res.label.map(lambda x: int(x))

# In[ ]:


res.to_csv('../result/lgb-baseline2.csv', index=False, header=False, sep=',', columns=['uid', 'label'])

