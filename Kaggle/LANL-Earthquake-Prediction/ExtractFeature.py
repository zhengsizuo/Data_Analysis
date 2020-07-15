# -*- coding: utf-8 -*-
'''
@Time    : 2019/5/21 9:38
@Author  : Zihao Huang
@File    : ExtractFeature
@Software: PyCharm
'''
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from scipy import stats
import scipy.signal as sg
from tsfresh.feature_extraction import feature_calculators

NY_FREQ_IDX = 75000  # the test signals are 150k samples long, Nyquist is thus 75k.
CUTOFF = 18000
MAX_FREQ_IDX = 20000
FREQ_STEP = 2500


def des_bw_filter_lp(cutoff=CUTOFF):  # low pass filter
    b, a = sg.butter(4, Wn=cutoff / NY_FREQ_IDX)
    return b, a


def feature_extract(X_train, i, X_element,y_train=None,y_element=None,is_TrainDataSet=True):
    if is_TrainDataSet:
        y_train.loc[i, 'time_to_failure'] = y_element

    X_element = X_element.reshape(-1)

    xcdm = X_element - np.mean(X_element)

    b, a = des_bw_filter_lp(cutoff=18000)
    xcz = sg.lfilter(b, a, xcdm)

    zc = np.fft.fft(xcz)
    zc = zc[:MAX_FREQ_IDX]

    # FFT transform values
    realFFT = np.real(zc)
    imagFFT = np.imag(zc)

    freq_bands = [x for x in range(0, MAX_FREQ_IDX, FREQ_STEP)]
    magFFT = np.sqrt(realFFT ** 2 + imagFFT ** 2)
    phzFFT = np.arctan(imagFFT / realFFT)
    phzFFT[phzFFT == -np.inf] = -np.pi / 2.0
    phzFFT[phzFFT == np.inf] = np.pi / 2.0
    phzFFT = np.nan_to_num(phzFFT)

    for freq in freq_bands:
        X_train.loc[i, 'FFT_Mag_01q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.01)
        X_train.loc[i, 'FFT_Mag_10q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.1)
        X_train.loc[i, 'FFT_Mag_90q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.9)
        X_train.loc[i, 'FFT_Mag_99q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.99)
        X_train.loc[i, 'FFT_Mag_mean%d' % freq] = np.mean(magFFT[freq: freq + FREQ_STEP])
        X_train.loc[i, 'FFT_Mag_std%d' % freq] = np.std(magFFT[freq: freq + FREQ_STEP])
        X_train.loc[i, 'FFT_Mag_max%d' % freq] = np.max(magFFT[freq: freq + FREQ_STEP])

        X_train.loc[i, 'FFT_Phz_mean%d' % freq] = np.mean(phzFFT[freq: freq + FREQ_STEP])
        X_train.loc[i, 'FFT_Phz_std%d' % freq] = np.std(phzFFT[freq: freq + FREQ_STEP])

    X_train.loc[i, 'FFT_Rmean'] = realFFT.mean()
    X_train.loc[i, 'FFT_Rstd'] = realFFT.std()
    X_train.loc[i, 'FFT_Rmax'] = realFFT.max()
    X_train.loc[i, 'FFT_Rmin'] = realFFT.min()
    X_train.loc[i, 'FFT_Imean'] = imagFFT.mean()
    X_train.loc[i, 'FFT_Istd'] = imagFFT.std()
    X_train.loc[i, 'FFT_Imax'] = imagFFT.max()
    X_train.loc[i, 'FFT_Imin'] = imagFFT.min()

    X_train.loc[i, 'FFT_Rmean_first_6000'] = realFFT[:6000].mean()
    X_train.loc[i, 'FFT_Rstd__first_6000'] = realFFT[:6000].std()
    X_train.loc[i, 'FFT_Rmax_first_6000'] = realFFT[:6000].max()
    X_train.loc[i, 'FFT_Rmin_first_6000'] = realFFT[:6000].min()
    X_train.loc[i, 'FFT_Rmean_first_18000'] = realFFT[:18000].mean()
    X_train.loc[i, 'FFT_Rstd_first_18000'] = realFFT[:18000].std()
    X_train.loc[i, 'FFT_Rmax_first_18000'] = realFFT[:18000].max()
    X_train.loc[i, 'FFT_Rmin_first_18000'] = realFFT[:18000].min()

    peaks = [10, 20, 50, 100]
    for peak in peaks:
        X_train.loc[i, 'num_peaks_{}'.format(peak)] = feature_calculators.number_peaks(X_element, peak)

    autocorr_lags = [5, 10, 50, 100, 500, 1000, 5000, 10000]
    for autocorr_lag in autocorr_lags:
        X_train.loc[i, 'autocorrelation_{}'.format(autocorr_lag)] = feature_calculators.autocorrelation(X_element,
                                                                                                  autocorr_lag)
        X_train.loc[i, 'c3_{}'.format(autocorr_lag)] = feature_calculators.c3(X_element, autocorr_lag)

    X_train.loc[i, 'ave'] = X_element.mean()
    X_train.loc[i, 'std'] = X_element.std()
    X_train.loc[i, 'max'] = X_element.max()
    X_train.loc[i, 'min'] = X_element.min()

    # geometric and harminic means
    X_train.loc[i, 'hmean'] = stats.hmean(np.abs(X_element[np.nonzero(X_element)[0]]))
    X_train.loc[i, 'gmean'] = stats.gmean(np.abs(X_element[np.nonzero(X_element)[0]]))

    # nth k-statistic and nth moment
    for ii in range(1, 5):
        X_train.loc[i, 'kstat_{}'.format(ii)] = stats.kstat(X_element, ii)
        X_train.loc[i, 'moment_{}'.format(ii)] = stats.moment(X_element, ii)

    for ii in [1, 2]:
        X_train.loc[i, 'kstatvar_{}.format(ii)'] = stats.kstatvar(X_element, ii)

    X_train.loc[i, 'max_to_min'] = X_element.max() / np.abs(X_element.min())
    X_train.loc[i, 'max_to_min_diff'] = X_element.max() - np.abs(X_element.min())
    X_train.loc[i, 'count_big'] = len(X_element[np.abs(X_element) > 500])
    X_train.loc[i, 'sum'] = X_element.sum()

    X_train.loc[i, 'av_change_abs'] = np.mean(np.diff(X_element))

    tmp=np.diff(X_element)/X_element[:-1]
    tmp = tmp[~np.isnan(tmp)]
    tmp = tmp[~np.isinf(tmp)]
    X_train.loc[i, 'av_change_rate'] = np.mean(tmp)

    X_train.loc[i, 'abs_max'] = np.abs(X_element).max()
    X_train.loc[i, 'abs_min'] = np.abs(X_element).min()

    X_train.loc[i, 'std_first_50000'] = X_element[:50000].std()
    X_train.loc[i, 'std_last_50000'] = X_element[-50000:].std()
    X_train.loc[i, 'std_first_10000'] = X_element[:10000].std()
    X_train.loc[i, 'std_last_10000'] = X_element[-10000:].std()

    X_train.loc[i, 'avg_first_50000'] = X_element[:50000].mean()
    X_train.loc[i, 'avg_last_50000'] = X_element[-50000:].mean()
    X_train.loc[i, 'avg_first_10000'] = X_element[:10000].mean()
    X_train.loc[i, 'avg_last_10000'] = X_element[-10000:].mean()

    X_train.loc[i, 'min_first_50000'] = X_element[:50000].min()
    X_train.loc[i, 'min_last_50000'] = X_element[-50000:].min()
    X_train.loc[i, 'min_first_10000'] = X_element[:10000].min()
    X_train.loc[i, 'min_last_10000'] = X_element[-10000:].min()

    X_train.loc[i, 'max_first_50000'] = X_element[:50000].max()
    X_train.loc[i, 'max_last_50000'] = X_element[-50000:].max()
    X_train.loc[i, 'max_first_10000'] = X_element[:10000].max()
    X_train.loc[i, 'max_last_10000'] = X_element[-10000:].max()

    percentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
    for p in percentiles:
        X_train.loc[i, 'percentile_{}'.format(p)] = np.percentile(X_element, p)
        X_train.loc[i, 'abs_percentile_{}'.format(p)] = np.percentile(np.abs(X_element), p)

    windows = [10, 50, 100, 500, 1000, 10000]
    X_element_df = pd.DataFrame(X_element)
    for w in windows:
        x_roll_std = X_element_df.rolling(w).std().dropna().values
        x_roll_mean = X_element_df.rolling(w).mean().dropna().values
        x_roll_std=x_roll_std.reshape(-1)
        x_roll_mean=x_roll_mean.reshape(-1)

        X_train.loc[i, 'ave_roll_std_{}'.format(w)] = x_roll_std.mean()
        X_train.loc[i, 'std_roll_std_{}'.format(w)] = x_roll_std.std()
        X_train.loc[i, 'max_roll_std_{}'.format(w)] = x_roll_std.max()
        X_train.loc[i, 'min_roll_std_{}'.format(w)] = x_roll_std.min()

        for p in percentiles:
            X_train.loc[i, 'percentile_roll_std_{}_window_{}'.format(p,w)] = np.percentile(x_roll_std, p)

        X_train.loc[i, 'av_change_abs_roll_std_{}'.format(w)] = np.mean(np.diff(x_roll_std))

        tmp = np.diff(x_roll_std) / x_roll_std[:-1]
        tmp = tmp[~np.isnan(tmp)]
        tmp = tmp[~np.isinf(tmp)]
        X_train.loc[i, 'av_change_rate_roll_std_{}'.format(w)] = np.mean(tmp)
        X_train.loc[i, 'abs_max_roll_std_{}'.format(w)] = np.abs(x_roll_std).max()

        X_train.loc[i, 'ave_roll_mean_{}'.format(w)] = x_roll_mean.mean()
        X_train.loc[i, 'std_roll_mean_{}'.format(w)] = x_roll_mean.std()
        X_train.loc[i, 'max_roll_mean_{}'.format(w)] = x_roll_mean.max()
        X_train.loc[i, 'min_roll_mean_{}'.format(w)] = x_roll_mean.min()

        for p in percentiles:
            X_train.loc[i, 'percentile_roll_mean_{}_window_{}'.format(p,w)] = np.percentile(x_roll_mean, p)

        X_train.loc[i, 'av_change_abs_roll_mean_{}'.format(w)] = np.mean(np.diff(x_roll_mean))

        tmp = np.diff(x_roll_mean) / x_roll_mean[:-1]
        tmp = tmp[~np.isnan(tmp)]
        tmp = tmp[~np.isinf(tmp)]
        X_train.loc[i, 'av_change_rate_roll_mean_{}'.format(w)] = np.mean(tmp)
        X_train.loc[i, 'abs_max_roll_mean_{}'.format(w)] = np.abs(x_roll_mean).max()




print('start to read train data')
train = pd.read_csv("./input/train.csv", dtype={"acoustic_data": np.int16, "time_to_failure": np.float32})
print('successfully read train data')


# 把训练部分分为5份： 0 - 3, 4 - 6, 7 - 9, 10 - 12, 13 - 16.用于交叉验证
sample_num = 300
input_len = 150000
np.random.seed(7898)

ttf = train["time_to_failure"].values
index_start = np.nonzero(np.diff(ttf) > 0)[0] + 1
index_start = np.insert(index_start, 0, 0)  # 插入起始index
chunk_length = np.diff(np.append(index_start, train.shape[0]))

X_series = []
y_series = []

cv_assign = [4, 7, 10, 13]
X, y = None, None
for i in range(len(index_start)):
    if i in cv_assign:
        X_series.append(X)
        y_series.append(y)
        X, y = None, None
    index_set = np.random.randint(low=index_start[i], high=index_start[i] + chunk_length[i] - input_len,
                                  size=sample_num)
    ac_data = np.zeros((sample_num, input_len, 1), dtype=np.int16)
    ac_label = np.zeros((sample_num,), dtype=np.float32)
    for j in range(sample_num):
        ac_data[j, :, 0] = train["acoustic_data"].values[index_set[j]:index_set[j] + input_len]
        ac_label[j] = train["time_to_failure"].values[index_set[j] + input_len]
    if X is None:
        X, y = ac_data, ac_label
    else:
        X, y = np.concatenate((X, ac_data), axis=0), np.concatenate((y, ac_label), axis=0)
X_series.append(X)
y_series.append(y)

X_series_feature = []
y_series_feature = []
print('start extract feature')


for k, X_batch in enumerate(X_series):
    X_train = pd.DataFrame(index=range(sample_num), dtype=np.float32)
    y_train = pd.DataFrame(index=range(sample_num), dtype=np.float32,
                           columns=['time_to_failure'])

    for i, X_element in enumerate(X_batch):
        feature_extract(X_train, i, X_element,y_train, y_series[k][i])
        print('train data processing: {}/{}, batch:{}/{}'.format(i+1,len(X_batch),k+1,len(X_series)))



    X_series_feature.append(X_train)
    y_series_feature.append(y_train)

#将训练集的特征存储到5个csv，方便交叉验证和回归
X_series_feature[0].to_csv("./input/X_series_feature_0.csv", index=True)
y_series_feature[0].to_csv("./input/y_series_feature_0.csv", index=True)

X_series_feature[1].to_csv("./input/X_series_feature_1.csv", index=True)
y_series_feature[1].to_csv("./input/y_series_feature_1.csv", index=True)

X_series_feature[2].to_csv("./input/X_series_feature_2.csv", index=True)
y_series_feature[2].to_csv("./input/y_series_feature_2.csv", index=True)

X_series_feature[3].to_csv("./input/X_series_feature_3.csv", index=True)
y_series_feature[3].to_csv("./input/y_series_feature_3.csv", index=True)

X_series_feature[4].to_csv("./input/X_series_feature_4.csv", index=True)
y_series_feature[4].to_csv("./input/y_series_feature_4.csv", index=True)


#提取测试集的特征
submission = pd.read_csv('./input/sample_submission.csv', index_col='seg_id')

X_test = pd.DataFrame(dtype=np.float32, index=submission.index)

for i, seg_id in enumerate(X_test.index):
    seg = pd.read_csv('./input/test/' + seg_id + '.csv')

    x = seg['acoustic_data'].values

    feature_extract(X_test,  seg_id, x, is_TrainDataSet=False)
    print('test data processing: {}/{}'.format( i+1, len(X_test.index)))

print('successfully extract feature')

X_test.to_csv("./input/X_test.csv", index=True)


