# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 14:26:57 2018

@author: NanguangZhou
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


def f1_score(preds, dtrain, threshold=0.367):
    labels = dtrain.get_label()
    # preds = 1 - preds
    preds = preds.copy()
    thres = threshold
    positive_box = preds>=thres
    negative_box = preds<thres
    preds[positive_box] = 1
    preds[negative_box] = 0
    
    con_mat = confusion_matrix(y_true=labels, y_pred=preds)
    p = con_mat[1, 1] / (con_mat[1, 1] + con_mat[0, 1])
    r = con_mat[1, 1] / (con_mat[1, 1] + con_mat[1, 0])
    # auc = roc_auc_score(labels, preds)
    #print('precision:', precision)
    #print('recall:' , recall)
    return 'f1_score',p,r, (2*p*r)/(p+r)


# 读取数据


test_data_a = pd.read_csv('../data/f_test_a_20180204.csv', low_memory=False, encoding='gbk')
test_data_a_answer = pd.read_csv('../data/f_answer_a_20180306.csv', low_memory=False, encoding='gbk', header=None)
test_data_a_answer.columns = ['label']
test_data_a['label'] = test_data_a_answer.label
data_origin = pd.read_csv('../data/f_train_20180204.csv', low_memory=False, encoding='gbk')
data_all = pd.concat([data_origin, test_data_a], axis=0)
columns = data_all.columns
data_all = pd.DataFrame(np.array(data_all))
data_all.columns = columns 
# data_all = pd.read_csv('d_train_20180102.csv', low_memory=False, encoding='gbk')
# test_data = pd.read_csv('f_test_a_20180204.csv', low_memory=False, encoding='gbk')
test_data = pd.read_csv('../data/f_test_b_20180305.csv', low_memory=False, encoding='gbk')


# 读取数据信息
from xml.dom import minidom
dom = minidom.parse('f_train_20180204_final.xml')
config = dom.getElementsByTagName('config')
b = config[0]
configs = {}
for item in b.getElementsByTagName('field'):
    configs[item.getAttribute('name')] = item.getAttribute('value')
print(configs)

# 记录入模变量
# 记录入模变量类型

from dateutil.parser import parse
feature_type = {}
data_dictionary = {}
## fileds 示例 <field binning_num="0" binning_type="percentile" m_type="字符型" missing="" name="NEW_QUES_DEL9" type="string" zh_name="手机里保存电话号码数"/>
fields = dom.getElementsByTagName('data')
b = fields[0]
string_columns = []
number_columns = []
columns = []
for item in b.getElementsByTagName('field'):
    col = item.getAttribute('name')
    col_type = item.getAttribute('type')
    data_dictionary[col] = item.getAttribute('zh_name')
    feature_type[col] = col_type
    bin_num = int(item.getAttribute('binning_num'))
    if col_type == 'string':
        string_columns.append(col)
    elif col_type == 'double':
        number_columns.append(col)
    elif col_type == 'date':
        print(1)
        data_all[col] = (pd.to_datetime(data_all[col]) - parse('2017-10-09')).dt.days
        test_data[col] = (pd.to_datetime(test_data[col]) - parse('2017-10-09')).dt.days
        feature_type[col] = 'double'
        number_columns.append(col)
    columns.append(col)


# 对字符型特征进行label编码, 在用xgboost LR SVM等其他不支持离散特征的模型需要进行Label OneHot
from sklearn.preprocessing import LabelEncoder
train_feat = data_all[columns].copy()
test_feat = test_data[columns].copy()
for col in string_columns:
    lab_enc = LabelEncoder()
    f_data = train_feat[col].apply(lambda x: str(x))
    t_data = test_feat[col].apply(lambda x: str(x))
    values_arr = list(f_data.unique())
    for val in t_data.unique():
        if val not in values_arr:
            values_arr.append(val)
    lab_enc.fit(values_arr)
    train_feat[col] = lab_enc.transform(f_data)
    test_feat[col] = lab_enc.transform(t_data)


# catboost cv 5折交叉验证
import catboost
if catboost.__version__ > '0.2.2':
    raise 'catboost版本不匹配， 请安装0.2.2版本，否则线上线下结果不匹配'
    
from sklearn.cross_validation import KFold
import time
from sklearn.metrics import roc_auc_score
cat_feature_inds = []
descreate_max_num = 20
for i, c in enumerate(columns):
    num_uniques = len(data_all[c].unique())
    if num_uniques < descreate_max_num:
        cat_feature_inds.append(i)

train_feat = data_all[columns].fillna(-999)#data_all[columns].median(axis=0))
test_feat = test_data[columns].fillna(-999)#test_data[columns].median(axis=0))

## 
print('开始CV 5折训练...')
t0 = time.time()
train_preds_cat = np.zeros(train_feat.shape[0])
test_preds_cat = np.zeros((test_feat.shape[0], 5))

kf = KFold(len(train_feat), n_folds=5, shuffle=True, random_state=descreate_max_num*26+1)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i+1))
    
    cat_model = catboost.CatBoostClassifier( 
        iterations = descreate_max_num*40, 
        learning_rate=0.03,
        depth=6, 
        l2_leaf_reg=1, 
        random_seed=i*100+6,
    )
    train_feat1 = train_feat.iloc[train_index]
    train_feat2 = train_feat.iloc[test_index]
    train_target1 = data_all.iloc[train_index]['label']
    train_target2 = data_all.iloc[test_index]['label']
    cat_model.fit(train_feat1, train_target1, cat_features=cat_feature_inds)
    test_preds_cat[:,i] = cat_model.predict_proba(test_feat[columns])[:,1]
    train_preds_cat[test_index] += cat_model.predict_proba(train_feat2)[:,1]
#    print('训练auc', roc_auc_score(train_target1, cat_model.predict_proba(train_feat1)[:,1]))
    print('test auc', roc_auc_score(train_target2, cat_model.predict_proba(train_feat2)[:,1]))

print('线下得分：    {}'.format(roc_auc_score(data_all['label'],train_preds_cat)))
print('CV训练用时{}秒'.format(time.time() - t0))
    

import xgboost as xgb
print('线下f1_score得分:{}'.format(
        f1_score(dtrain=xgb.DMatrix(train_feat.values, 
        label=data_all['label']),
        preds=train_preds_cat,threshold=0.372)))

pd.Series(test_preds_cat.mean(axis=1)).to_csv('predict_prob_test.txt', index=False,float_format='%.4f',header=False )

print('选取45%左右的正样本的概率值作为阈值')
# 根据前两次60%成绩，设定阈值为 0.4 103个正样本
result_cat = test_preds_cat.mean(axis=1)
result_cat[result_cat>=0.4] = 1
result_cat[result_cat<0.4] = 0
result_cat = pd.Series(result_cat)
print(result_cat.value_counts())


import datetime
result_cat.to_csv(("../submit/submit_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"), header=None, index=False)

