
from __future__ import print_function
from __future__ import division
from sklearn import metrics
from matplotlib import pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')
import xgboost as xgb
import pandas as pd
import numpy as np


# load the provided data
train_features = pd.read_csv('/home/bugra/PycharmProjects/bil470_project/dengue_features_train.csv',index_col=[0,1,2])

train_labels = pd.read_csv('/home/bugra/PycharmProjects/bil470_project/dengue_labels_train.csv',index_col=[0,1,2])


def preprocess_data(data_path, labels_path=None):

    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])

    # select features we want
    features =  ['reanalysis_precip_amt_kg_per_m2', 'reanalysis_air_temp_k', 'precipitation_amt_mm', 'station_precip_mm']
    df = df[features]


    # fill missing values
    df = df.fillna(df.mean()) # 6.49


    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        df = df.join(labels)

    # separate san juan and iquitos
    sj = df.loc['sj']
    iq = df.loc['iq']

    return sj, iq

sj_train, iq_train = preprocess_data('/home/bugra/PycharmProjects/bil470_project/dengue_features_train.csv',labels_path="/home/bugra/PycharmProjects/bil470_project/dengue_labels_train.csv")
sj_train_subtrain = sj_train.head(800)
sj_train_subtest = sj_train.tail(sj_train.shape[0] - 800)

iq_train_subtrain = iq_train.head(400)
iq_train_subtest = iq_train.tail(iq_train.shape[0] - 400)
print("XGBoost regression")

X=sj_train_subtrain[ ['reanalysis_precip_amt_kg_per_m2', 'reanalysis_air_temp_k', 'precipitation_amt_mm', 'station_precip_mm']]
Y=sj_train_subtrain['total_cases']

X_test=sj_train_subtest[ ['reanalysis_precip_amt_kg_per_m2', 'reanalysis_air_temp_k', 'precipitation_amt_mm', 'station_precip_mm']]
Y_test=sj_train_subtest['total_cases']

dtrain=xgb.DMatrix(X,label=Y)
dtest=xgb.DMatrix(X_test,label=Y_test)

#bst=xgb.XGBRegressor() mae:27.953230

bst=xgb.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,
                 learning_rate=0.05,
                 max_depth=5,
                 min_child_weight=1,
                 n_estimators=9,
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.4,
                 seed=0)    # mae 15.994096636772156   """"""

bst.fit(X,Y)
preds=bst.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, preds))
df=pd.DataFrame({'Actual': Y_test, 'Predicted': preds})
df.plot(kind='line',figsize=(20,8))
plt.grid(which='major', linestyle='-', linewidth='0.1', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.1', color='black')
plt.show()




#iq
X_iq=iq_train_subtrain[ ['reanalysis_precip_amt_kg_per_m2', 'reanalysis_air_temp_k', 'precipitation_amt_mm', 'station_precip_mm']]
Y_iq=iq_train_subtrain['total_cases']

X_test_iq=iq_train_subtest[ ['reanalysis_precip_amt_kg_per_m2', 'reanalysis_air_temp_k', 'precipitation_amt_mm', 'station_precip_mm']]
Y_test_iq=iq_train_subtest['total_cases']

dtrain_iq=xgb.DMatrix(X_iq,label=Y_iq)
dtest_iq=xgb.DMatrix(X_test_iq,label=Y_test_iq)

#bst=xgb.XGBRegressor() mae:27.953230

bst_iq=xgb.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,
                 learning_rate=0.1,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=9,
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.4,
                 seed=40)    # mae 15.994096636772156   """"""

bst_iq.fit(X_iq,Y_iq)
preds_iq=bst_iq.predict(X_test_iq)
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test_iq, preds_iq))
df_iq=pd.DataFrame({'Actual': Y_test_iq, 'Predicted': preds_iq})
df_iq.plot(kind='line',figsize=(20,8))
plt.grid(which='major', linestyle='-', linewidth='0.1', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.1', color='black')
plt.show()

#print("colsample_bytree=0.4\ngamma=0\nlearning_rate=0.1\nmax_depth=3\nmin_child_weight=1.5\nn_estimators=9\nreg_alpha=0.75\nreg_lambda=0.45\nsubsample=0.4\nseed=40")

sj_test, iq_test = preprocess_data('/home/bugra/PycharmProjects/bil470_project/dengue_features_test.csv')

sj_predictions = bst.predict(sj_test).astype(int)
iq_predictions = bst_iq.predict(iq_test).astype(int)
submission = pd.read_csv("/home/bugra/PycharmProjects/bil470_project/submission_format.csv",
                         index_col=[0, 1, 2])

submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
submission.to_csv("/home/bugra/PycharmProjects/bil470_project/benchmark.csv")
