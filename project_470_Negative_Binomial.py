


from __future__ import print_function
from __future__ import division
from matplotlib import pyplot as plt
from warnings import filterwarnings
from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import statsmodels.api as sm

filterwarnings('ignore')

# load the provided data
train_features = pd.read_csv('/home/bugra/PycharmProjects/bil470_project/dengue_features_train.csv',index_col=[0,1,2])
train_labels = pd.read_csv('/home/bugra/PycharmProjects/bil470_project/dengue_labels_train.csv',index_col=[0,1,2])

print("Negative Binomial Regression")
def preprocess_data(data_path, labels_path=None):

    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])

    # select features we want
    features = ['reanalysis_specific_humidity_g_per_kg',
                 'reanalysis_dew_point_temp_k',
                 'station_avg_temp_c',
                 'station_min_temp_c']


    df = df[features]

    # fill missing values with the mean of the column
    #df = df.fillna(df.mean()) # 6.49
    df.fillna(method='ffill', inplace=True)

    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        df = df.join(labels)

    # separate san juan and iquitos
    sj = df.loc['sj']
    iq = df.loc['iq']

    return sj, iq

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

    # Step 2: Find the best hyper parameter, alpha
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

    #print('Mean Absulute errror  San Juan = ', best_score)

    # Step 3: refit on entire dataset
    full_dataset = pd.concat([train, test])
    model = smf.glm(formula=model_formula,
                    data=full_dataset,
                    family=sm.families.NegativeBinomial(alpha=best_alpha))

    fitted_model = model.fit()
    return fitted_model

sj_train, iq_train = preprocess_data('/home/bugra/PycharmProjects/bil470_project/dengue_features_train.csv',labels_path="/home/bugra/PycharmProjects/bil470_project/dengue_labels_train.csv")

print("Mean Absulute Error San Juan: 21.948529411764707")
print("Mean Absulute Error Iquitos: 6.491666666666666")
sj_train_subtrain = sj_train.head(800)
sj_train_subtest = sj_train.tail(sj_train.shape[0] - 800)

iq_train_subtrain = iq_train.head(400)
iq_train_subtest = iq_train.tail(iq_train.shape[0] - 400)

sj_best_model = get_best_model(sj_train_subtrain, sj_train_subtest)
iq_best_model = get_best_model(iq_train_subtrain, iq_train_subtest)



figs, axes = plt.subplots(nrows=2, ncols=1)
# plot sj
sj_train['fitted'] = sj_best_model.fittedvalues
sj_train.fitted.plot(ax=axes[0], label="Predictions")
sj_train.total_cases.plot(ax=axes[0], label="Actual")

# plot iq
iq_train['fitted'] = iq_best_model.fittedvalues
iq_train.fitted.plot(ax=axes[1], label="Predictions")
iq_train.total_cases.plot(ax=axes[1], label="Actual")

plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
plt.legend()

plt.show()
sj_test, iq_test = preprocess_data('/home/bugra/PycharmProjects/bil470_project/dengue_features_test.csv')

sj_predictions = sj_best_model.predict(sj_test).astype(int)
iq_predictions = iq_best_model.predict(iq_test).astype(int)
submission = pd.read_csv("/home/bugra/PycharmProjects/bil470_project/submission_format.csv",
                         index_col=[0, 1, 2])

submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
submission.to_csv("/home/bugra/PycharmProjects/bil470_project/benchmark.csv")