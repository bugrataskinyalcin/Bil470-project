
from __future__ import print_function
from __future__ import division

from matplotlib.pyplot import axes
from sklearn import metrics
from matplotlib import pyplot as plt
from warnings import filterwarnings
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.tools import eval_measures

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
    features = ['reanalysis_specific_humidity_g_per_kg',
                 'reanalysis_dew_point_temp_k',
                 'station_avg_temp_c',
                 'station_min_temp_c']
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


X=sj_train_subtrain[ ['reanalysis_specific_humidity_g_per_kg',
                 'reanalysis_dew_point_temp_k',
                 'station_avg_temp_c',
                 'station_min_temp_c']]
Y=sj_train_subtrain['total_cases']
print("XGBoost and Negative Binomial Regressions")
X_test=sj_train_subtest[ ['reanalysis_specific_humidity_g_per_kg',
                 'reanalysis_dew_point_temp_k',
                 'station_avg_temp_c',
                 'station_min_temp_c']]
Y_test=sj_train_subtest['total_cases']

dtrain=xgb.DMatrix(X,label=Y)
dtest=xgb.DMatrix(X_test,label=Y_test)

#bst=xgb.XGBRegressor() mae:27.953230

bst=xgb.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,
                 learning_rate=0.05,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=9,
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.4,
                 seed=40)    # mae 15.994096636772156   """"""

bst.fit(X,Y)
preds=bst.predict(X_test)
print('Mean Absolute Error San Juan:', metrics.mean_absolute_error(Y_test, preds))
df=pd.DataFrame({'Actual': Y_test, 'Predicted': preds})
df.plot(kind='line',figsize=(20,8))
plt.grid(which='major', linestyle='-', linewidth='0.1', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.1', color='black')
plt.show()


def get_best_model(train, test):
    # Step 1: specify the form of the model
    model_formula = "total_cases ~ 1 + " \
                    "reanalysis_specific_humidity_g_per_kg + " \
                    "reanalysis_dew_point_temp_k + " \
                    "station_min_temp_c + " \
                    "station_avg_temp_c"


    grid = 10 ** np.arange(-8, -3, dtype=np.float64)

    best_alpha = []
    best_score = 1000

    # Step 2: Find the best hyper parameter, alpha μ+αμ2.
    for alpha in grid:
        model = smf.glm(formula=model_formula,
                        data=train,
                        family=sm.families.NegativeBinomial(alpha=alpha))

        results = model.fit()
        predictions = results.predict(test).astype(int)
        score = eval_measures.meanabs(predictions, test.total_cases)

        if score < best_score:
            best_alpha = alpha
            best_score = score

    print('Mean Absolute Error Iquitos: = ', best_score)

    # Step 3: refit on entire dataset
    full_dataset = pd.concat([train, test])
    model = smf.glm(formula=model_formula,
                    data=full_dataset,
                    family=sm.families.NegativeBinomial(alpha=best_alpha))

    fitted_model = model.fit()
    return fitted_model

iq_best_model = get_best_model(iq_train_subtrain, iq_train_subtest)
sj_test, iq_test = preprocess_data('/home/bugra/PycharmProjects/bil470_project/dengue_features_test.csv')

sj_predictions = bst.predict(sj_test).astype(int)
iq_predictions = iq_best_model.predict(iq_test).astype(int)

submission = pd.read_csv("/home/bugra/PycharmProjects/bil470_project/submission_format.csv",
                         index_col=[0, 1, 2])

submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
submission.to_csv("/home/bugra/PycharmProjects/bil470_project/benchmark.csv")