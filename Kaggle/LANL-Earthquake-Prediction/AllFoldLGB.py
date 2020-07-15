# -*- coding: utf-8 -*-
'''
@Time    : 2019/5/23 20:12
@Author  : Zihao Huang
@File    : AllFoldLGB
@Software: PyCharm
'''

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


def train_model(X, X_test, y, folds, params=None, model_type='lgb', plot_feature_importance=False, model=None):
    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print('Fold', fold_n, 'started at', time.ctime())
        if type(X) == np.ndarray:
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators=1000, n_jobs=-1)
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
                      verbose=100, early_stopping_rounds=200)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                              verbose_eval=500, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns),
                                         ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid).reshape(-1, )
            score = mean_absolute_error(y_valid, y_pred_valid)
            print(f'Fold {fold_n}. MAE: {score:.4f}.')
            print('')

            y_pred = model.predict(X_test).reshape(-1, )

        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000, eval_metric='MAE', **params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True,
                      verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)

        oof[valid_index] = y_pred_valid.reshape(-1, )
        scores.append(mean_absolute_error(y_valid, y_pred_valid))

        prediction += y_pred

        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_fold

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    if model_type == 'lgb':
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12))
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
            plt.title('LGB Features (avg over folds)')
            plt.show()

            return oof, prediction, feature_importance
        return oof, prediction, scores

    else:
        return oof, prediction, scores

#将5个训练集文件中的样本全部混到一起
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


X=pd.concat(X_series_feature)
y=pd.concat(y_series_feature)

params = {'num_leaves': 21,
              'min_data_in_leaf': 10,
              'objective': 'regression',
              'learning_rate': 0.01,
              'max_depth': 108,
              "boosting": "gbdt",
              "feature_fraction": 0.8,
              "bagging_freq": 1,
              "bagging_fraction": 0.8,
              "bagging_seed": 42,
              "metric": 'mae',
              "lambda_l1": 0.1,
              "verbosity": -1,
              "random_state": 42}

n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)

all_importance=[]

#5折交叉验证
for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
    print('Fold', fold_n, 'started at', time.ctime())
    if type(X) == np.ndarray:
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
    else:
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    #标准化
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)

    X_valid = pd.DataFrame(scaler.transform(X_valid), columns=X_valid.columns)

    #训练模型
    model = lgb.LGBMRegressor(**params, n_estimators=1000, n_jobs=-1)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
              verbose=100, early_stopping_rounds=200)

    fold_importance = pd.DataFrame()
    fold_importance["feature"] = X_train.columns
    #记录本次模型特征重要程度
    fold_importance["importance"] = model.feature_importances_
    best_feature = fold_importance.sort_values(by='importance', ascending=False)

    all_importance.append(best_feature)

performance=pd.DataFrame()
y_pred_lgb = np.zeros((2624), dtype=np.float32)

X_test1=copy.deepcopy(X_test)

#特征选择
for rate in np.arange(0.1,1.1,0.1):
    rate=0.5#the best
    scores_val = []
    scores_fit=[]
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        if type(X) == np.ndarray:
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        bad_feature = all_importance[fold_n].feature[int(441 * rate):].values
        X_test=copy.deepcopy(X_test1)

        for feat in bad_feature:
            del X_train[feat]
            del X_valid[feat]
            del X_test[feat]


        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)

        X_valid = pd.DataFrame(scaler.transform(X_valid), columns=X_valid.columns)



        model = lgb.LGBMRegressor(**params, n_estimators=1000, n_jobs=-1)
        model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
                  verbose=100, early_stopping_rounds=200)

        preds = model.predict(X_valid, num_iteration=model.best_iteration_)
        mae = mean_absolute_error(y_valid, preds)

        preds = model.predict(X_train, num_iteration=model.best_iteration_)
        mae_fit = mean_absolute_error(y_train, preds)

        print('LightGBM--MAE:{}, Fold:{},  rate:{}'.format(mae, fold_n,rate))

        scores_val.append(mae)
        scores_fit.append(mae_fit)

        #预测测试集数据
        y_pred_lgb+= model.predict(X_test, num_iteration=model.best_iteration_)

    performance.loc[int(rate * 10), 'rate'] = rate
    performance.loc[int(rate * 10), 'fit mean'] = np.mean(scores_fit)
    performance.loc[int(rate * 10), 'fit std'] = np.std(scores_fit)
    performance.loc[int(rate * 10), 'val mean'] = np.mean(scores_val)
    performance.loc[int(rate * 10), 'val std'] = np.std(scores_val)
    break




print(performance)

submission.time_to_failure = y_pred_lgb/5
submission.to_csv("submission.csv", index=True)


