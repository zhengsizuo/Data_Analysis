# -*- coding: utf-8 -*-
'''
@Time    : 2019/5/21 9:38
@Author  : Zihao Huang
@File    : LGB_regression
@Software: PyCharm
'''
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
import time
import seaborn as sns
import matplotlib.pyplot as plt
import copy
from sklearn.feature_selection import SelectKBest, f_regression

# 读取特征
X_series_feature = []
y_series_feature = []
for i in range(5):
    train = pd.read_csv("./input/X_series_feature_{}.csv".format(i))
    label = pd.read_csv("./input/y_series_feature_{}.csv".format(i))

    del train['Unnamed: 0']
    del label['Unnamed: 0']

    X_series_feature.append(train)
    y_series_feature.append(label)

submission = pd.read_csv('./input/sample_submission.csv', index_col='seg_id')

X_test = pd.read_csv("./input/X_test.csv")
del X_test['seg_id']

y_pred_lgb = np.zeros((2624), dtype=np.float32)
y_pred_xgb = np.zeros((2624), dtype=np.float32)
lgb_model_loss = np.zeros((5,), dtype=np.float32)
xgb_model_loss = np.zeros((5,), dtype=np.float32)
#交叉验证
for i in range(len(X_series_feature)):
    i = 3  # the best
    #生成训练集和测试集
    X_fit, y_fit = None, None
    X_val, y_val = None, None
    for j in range(len(X_series_feature)):
        if j != i:
            if X_fit is None:
                X_fit, y_fit = X_series_feature[j], y_series_feature[j]
            else:
                X_fit, y_fit = pd.concat([X_fit, X_series_feature[j]], axis=0), pd.concat(
                    [y_fit, y_series_feature[j]],
                    axis=0)
    X_val, y_val = X_series_feature[i], y_series_feature[i]

    #基于训练集数据对所有数据标准化
    scaler = StandardScaler()
    scaler.fit(X_fit)
    X_fit_scaled = pd.DataFrame(scaler.transform(X_fit), columns=X_fit.columns)

    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    params = {'num_leaves': 21,
              'min_data_in_leaf': 10,
              'objective': 'regression',
              'learning_rate': 0.005,
              'max_depth': 50,
              "boosting": "gbdt",
              "feature_fraction": 0.8,
              "bagging_freq": 1,
              "bagging_fraction": 0.8,
              "bagging_seed": 42,
              "metric": 'mae',
              "lambda_l1": 0.1,
              "verbosity": -1,
              "random_state": 42}  # MAE:1.38622

    #训练模型
    model = lgb.LGBMRegressor(**params, n_estimators=1100, n_jobs=-1)
    model.fit(X_fit_scaled, y_fit, verbose=100,
              eval_set=[(X_fit_scaled, y_fit), (X_val_scaled, y_val)], eval_metric='mae')
    y_pred_lgb += model.predict(X_test_scaled, num_iteration=model.best_iteration_)
    preds = model.predict(X_val_scaled, num_iteration=model.best_iteration_)
    mae = mean_absolute_error(y_val, preds)
    print('LightGBM--Fold {}: MAE:{}'.format(i, mae))
    lgb_model_loss[i] = mae
    break

submission.time_to_failure = y_pred_lgb
submission.to_csv("submission.csv", index=True)

X_fit_scaled

# 特征选择
# fold_importance = pd.DataFrame()
# fold_importance["feature"] = X_fit_scaled.columns
# fold_importance["importance"] = model.feature_importances_
# fold_importance.to_csv("fold_importance.csv", index=False)
#
# best_feature = fold_importance.sort_values(by='importance', ascending=False) #根据前面训练的模型获得特征重要程度
#
# X_fit_scaled1 = copy.deepcopy(X_fit_scaled)
# X_val_scaled1 = copy.deepcopy(X_val_scaled)
# X_test_scaled1 = copy.deepcopy(X_test_scaled)
#
# performance = pd.DataFrame()
#
# for rate in np.arange(0.05, 1.05, 0.05):#按比例删除一定量的特征
#
#     rate=0.85 #the best
#
#     X_fit_scaled = copy.deepcopy(X_fit_scaled1)
#     X_val_scaled = copy.deepcopy(X_val_scaled1)
#     X_test_scaled = copy.deepcopy(X_test_scaled1)
#
#     bad_feature = best_feature.feature[int(441 * rate):].values
#
#     for feat in bad_feature:
#         del X_fit_scaled[feat]
#         del X_val_scaled[feat]
#         del X_test_scaled[feat]
#
#     model = lgb.LGBMRegressor(**params, n_estimators=1100, n_jobs=-1)
#     model.fit(X_fit_scaled, y_fit, verbose=100,
#               eval_set=[(X_fit_scaled, y_fit), (X_val_scaled, y_val)], eval_metric='mae')
#     y_pred_lgb = model.predict(X_test_scaled, num_iteration=model.best_iteration_)
#     preds = model.predict(X_val_scaled, num_iteration=model.best_iteration_)
#     mae = mean_absolute_error(y_val, preds)
#
#     preds = model.predict(X_fit_scaled, num_iteration=model.best_iteration_)
#     mae_fit = mean_absolute_error(y_fit, preds)
#
#     print('LightGBM--MAE:{}, rate:{}'.format(mae, rate))
#
#     performance.loc[int(rate * 20), 'rate'] = rate
#     performance.loc[int(rate * 20), 'mae fit'] = mae_fit
#     performance.loc[int(rate * 20), 'mae val'] = mae
#
#     submission.time_to_failure = y_pred_lgb
#     submission.to_csv("submission.csv", index=True)
#     break
#
#
# print(performance)
