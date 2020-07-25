
from __future__ import print_function
from __future__ import division

import matplotlib as matplotlib
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.impute import SimpleImputer

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# just for the sake of this blog post!
from warnings import filterwarnings
from sklearn import linear_model

from sklearn.preprocessing import PolynomialFeatures

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
sj_train_labels_subset=train_labels.head(800)


X=sj_train_subtrain[['reanalysis_specific_humidity_g_per_kg',
                'reanalysis_dew_point_temp_k',
                'station_avg_temp_c',
                'station_min_temp_c']]
Y=sj_train_subtrain['total_cases']

X_test=sj_train_subtest[['reanalysis_specific_humidity_g_per_kg',
                'reanalysis_dew_point_temp_k',
                'station_avg_temp_c',
                'station_min_temp_c']]
Y_test=sj_train_subtest['total_cases']


print("Polynomial Regression")

poly=PolynomialFeatures(degree=4,include_bias=True)  #24.365032988763073->4 ,22.03887342969829->1
X_poly=poly.fit_transform(X)
X_test_poly=poly.fit_transform(X_test)
model=LinearRegression(fit_intercept=False)
model.fit(X_poly,Y)
preds=model.predict(X_test_poly)
print('Mean Absolute Error San Juan:', metrics.mean_absolute_error(Y_test, preds))
df=pd.DataFrame({'Actual': Y_test, 'Predicted': preds})
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

PolynomialFeatures(degree=4,include_bias=True)  #24.365032988763073->4 ,22.03887342969829->1
X_poly_iq=poly.fit_transform(X_iq)
X_test_poly_iq=poly.fit_transform(X_test_iq)
model_iq=LinearRegression(fit_intercept=False)
model_iq.fit(X_poly_iq,Y_iq)
preds_iq=model_iq.predict(X_test_poly_iq)
print('Mean Absolute Error Iquitos:', metrics.mean_absolute_error(Y_test_iq, preds_iq))
df_iq=pd.DataFrame({'Actual': Y_test_iq, 'Predicted': preds_iq})
df_iq.plot(kind='line',figsize=(20,8))
plt.grid(which='major', linestyle='-', linewidth='0.1', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.1', color='black')
plt.show()

