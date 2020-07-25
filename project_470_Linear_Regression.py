
from __future__ import print_function
from __future__ import division
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

import pandas as pd


# just for the sake of this blog post!
from warnings import filterwarnings
filterwarnings('ignore')


# load the provided data
train_features = pd.read_csv('/home/bugra/PycharmProjects/bil470_project/dengue_features_train.csv',index_col=[0,1,2])

train_labels = pd.read_csv('/home/bugra/PycharmProjects/bil470_project/dengue_labels_train.csv',index_col=[0,1,2])


def preprocess_data(data_path, labels_path=None):

    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])

    # select features we want
    features = ['reanalysis_specific_humidity_g_per_kg',
                'reanalysis_dew_point_temp_k',
                'station_avg_temp_c',
                'station_min_temp_c']
    df = df[features]


    # fill missing values
    #df.fillna(method='ffill', inplace=True) #6.46
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


X_sj=sj_train_subtrain[['reanalysis_specific_humidity_g_per_kg',
                'reanalysis_dew_point_temp_k',
             'station_avg_temp_c',
                'station_min_temp_c']]
Y_sj=sj_train_subtrain['total_cases']

X_test_sj=sj_train_subtest[['reanalysis_specific_humidity_g_per_kg',
                'reanalysis_dew_point_temp_k',
                'station_avg_temp_c',
                'station_min_temp_c']]
Y_test_sj=sj_train_subtest['total_cases']

regressor=LinearRegression()
regressor.fit(X_sj,Y_sj)

coeff_df = pd.DataFrame(regressor.coef_, X_sj.columns, columns=['Coefficient'])

y_pred_sj=regressor.predict(X_test_sj)

df=pd.DataFrame({'Actual': Y_test_sj, 'Predicted': y_pred_sj})
print("Linear Regression")
print('Mean Absolute Error San Juan:', metrics.mean_absolute_error(Y_test_sj, y_pred_sj))

df.plot(kind='line',figsize=(20,8))
plt.grid(which='major', linestyle='-', linewidth='0.1', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.1', color='black')
plt.show()


X_iq=iq_train_subtrain[['reanalysis_specific_humidity_g_per_kg',
                'reanalysis_dew_point_temp_k',
             'station_avg_temp_c',
                'station_min_temp_c']]
Y_iq=iq_train_subtrain['total_cases']

X_test_iq=iq_train_subtest[['reanalysis_specific_humidity_g_per_kg',
                'reanalysis_dew_point_temp_k',
                'station_avg_temp_c',
                'station_min_temp_c']]
Y_test_iq=iq_train_subtest['total_cases']

regressor_iq=LinearRegression()
regressor_iq.fit(X_iq,Y_iq)

coeff_df = pd.DataFrame(regressor_iq.coef_, X_iq.columns, columns=['Coefficient'])

y_pred_iq=regressor_iq.predict(X_test_iq)

dfiq=pd.DataFrame({'Actual': Y_test_iq, 'Predicted': y_pred_iq})

print('Mean Absolute Error Iquitos:', metrics.mean_absolute_error(Y_test_iq, y_pred_iq))

dfiq.plot(kind='line',figsize=(20,8))
plt.grid(which='major', linestyle='-', linewidth='0.1', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.1', color='black')
plt.show()

sj_test, iq_test = preprocess_data('/home/bugra/PycharmProjects/bil470_project/dengue_features_train.csv')

sj_predictions = regressor.predict(sj_test).astype(int)
iq_predictions = regressor_iq.predict(iq_test).astype(int)

