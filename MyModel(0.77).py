
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import metrics
from sklearn import model_selection
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import Imputer
import lightgbm as lgb

# In[ ]:

uid_train = pd.read_csv('../data/uid_train.txt',sep='\t',header=None,names=('uid','label'))
voice_train = pd.read_csv('../data/voice_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':str,'end_time':str})
sms_train = pd.read_csv('../data/sms_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str})
wa_train = pd.read_csv('../data/wa_train.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str})


# In[ ]:


voice_test = pd.read_csv('../data/voice_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':str,'end_time':str})
sms_test = pd.read_csv('../data/sms_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str})
wa_test = pd.read_csv('../data/wa_test_b.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str})


# In[ ]:


uid_test = pd.DataFrame({'uid':pd.unique(wa_test['uid'])})
uid_test.to_csv('../data/uid_test_b.txt',index=None)


# In[ ]:


voice = pd.concat([voice_train,voice_test],axis=0)
sms = pd.concat([sms_train,sms_test],axis=0)
wa = pd.concat([wa_train,wa_test],axis=0)


# ##  baseline

# ### 通话记录

# In[ ]:


voice_opp_num = voice.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('voice_opp_num_').reset_index()

voice_opp_head=voice.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('voice_opp_head_').reset_index()

voice_opp_len=voice.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('voice_opp_len_').reset_index().fillna(0)

voice_call_type = voice.groupby(['uid','call_type'])['uid'].count().unstack().add_prefix('voice_call_type_').reset_index().fillna(0)

voice_in_out = voice.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('voice_in_out_').reset_index().fillna(0)


# ## 短信记录

# In[ ]:


sms_opp_num = sms.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('sms_opp_num_').reset_index()

sms_opp_head=sms.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('sms_opp_head_').reset_index()

sms_opp_len=sms.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('sms_opp_len_').reset_index().fillna(0)


sms_in_out = sms.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('sms_in_out_').reset_index().fillna(0)


# ### 网站/APP记录

# In[ ]:


wa_name = wa.groupby(['uid'])['wa_name'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('wa_name_').reset_index()
visit_cnt = wa.groupby(['uid'])['visit_cnt'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_visit_cnt_').reset_index()

visit_dura = wa.groupby(['uid'])['visit_dura'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_visit_dura_').reset_index()


up_flow = wa.groupby(['uid'])['up_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_up_flow_').reset_index()

down_flow = wa.groupby(['uid'])['down_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_down_flow_').reset_index()


# In[ ]:


feature = [voice_opp_num,voice_opp_head,voice_opp_len,voice_call_type,voice_in_out,sms_opp_num,sms_opp_head,sms_opp_len,sms_in_out,wa_name,visit_cnt,visit_dura,up_flow,
           down_flow]

'''
# In[ ]:
# 中文乱码处理
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 确保绘制的饼图为圆形
plt.axes(aspect='equal')

# 统计为风险用户的频数
counts = uid_train.label.value_counts()

# 绘制饼图
plt.pie(x=counts,
        labels=pd.Series(counts.index).map({0: 'zero', 1: 'one'}),
        autopct='%.2f%%')
plt.show()
'''


train_feature = uid_train
for feat in feature:
    train_feature = pd.merge(train_feature,feat,how='left',on='uid')
train_feature.fillna(0)

test_feature = uid_test
for feat in feature:
    test_feature = pd.merge(test_feature, feat, how='left', on='uid')

train = train_feature.drop(['uid', 'label'], axis=1)
x_train = Imputer().fit_transform(train.values)
# x_train = train_feature.drop(['uid', 'label'], axis=1)
y_train = train_feature.label

test = test_feature.drop(['uid'], axis=1)
x_test = Imputer().fit_transform(test.values)
# x_test = test_feature.drop(['uid'], axis=1)

### lgb
dtrain = lgb.Dataset(x_train, label=y_train)
dtest = lgb.Dataset(x_test)

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
#    'metric': ('multi_logloss', 'multi_error'),
    #'metric_freq': 100,
    'is_training_metric': False,
    'min_data_in_bin': 45,
    'num_leaves': 200,
    'learning_rate': 0.125,
    'max_bin': 600,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'verbosity': -1,
#    'gpu_device_id':2,
#    'device':'gpu'
#    'lambda_l1': 0.001,
#    'skip_drop': 0.95,
#    'max_drop' : 10
    #'lambda_l2': 0.005
    #'num_threads': 18
}


def evalMetric(preds, dtrain):
    label = dtrain.get_label()

    pre = pd.DataFrame({'preds': preds, 'label': label})
    pre = pre.sort_values(by='preds', ascending=False)

    auc = metrics.roc_auc_score(pre.label, pre.preds)

    pre.preds = pre.preds.map(lambda x: 1 if x >= 0.5 else 0)

    f1 = metrics.f1_score(pre.label, pre.preds)

    res = 0.6 * auc + 0.4 * f1

    return 'res', res, True

# ### 本地CV

# In[ ]:


lgb.cv(lgb_params,
       dtrain,
       feval=evalMetric,
       early_stopping_rounds=100,
       verbose_eval=5,
       num_boost_round=10000,
       nfold=3,
       metrics=['evalMetric'])

# ## 训练

# In[ ]:


model = lgb.train(lgb_params, dtrain, feval=evalMetric, verbose_eval=5, num_boost_round=300, valid_sets=[dtrain])


# ### 预测

# In[ ]:


# 对训练数据集作平衡处理
over_samples = SMOTE(random_state=1234)
over_samples_X, over_samples_y = over_samples.fit_sample(x_train, y_train)

# 重抽样前的类别比例
print(y_train.value_counts()/len(y_train))
# 重抽样后的类别比例
print(pd.Series(over_samples_y).value_counts()/len(over_samples_y))

dtrain2 = lgb.Dataset(over_samples_X, label=over_samples_y)
model2 = lgb.train(lgb_params, dtrain2, verbose_eval=5, num_boost_round=300, valid_sets=[dtrain2])

lgb.cv(lgb_params,
       dtrain2,
       feval=evalMetric,
       early_stopping_rounds=100,
       verbose_eval=5,
       num_boost_round=10000,
       nfold=3,
       metrics=['evalMetric'])

# ## 训练

# In[ ]:


model2 = lgb.train(lgb_params, dtrain2, feval=evalMetric, verbose_eval=5, num_boost_round=300, valid_sets=[dtrain2])

# ### 预测

# In[ ]:


pred = model2.predict(x_test)


# In[ ]:


res =pd.DataFrame({'uid':test_feature.uid, 'label':pred})


# In[ ]:


res=res.sort_values(by='label',ascending=False)
res.label=res.label.map(lambda x: 1 if x>=0.5 else 0)
res.label = res.label.map(lambda x: int(x))


# In[ ]:


res.to_csv('../result/finalB.csv', index=False, header=False, sep=',' ,columns=['uid', 'label'])



