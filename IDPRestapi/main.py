"""# Libraries"""

# Commented out IPython magic to ensure Python compatibility.
import opendatasets as od
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import os
# visualize data
from scipy.stats import norm
from scipy import stats
# Import missingno as msno
import missingno as msno
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import neural_network, linear_model, preprocessing, svm, tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'
"""# Model Configrations and functions"""

mse_values = list()
variance_score = list()

from sklearn.metrics import r2_score
from scipy import stats  # For in-built method to get PCC
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from scipy.stats import norm
import scikitplot as skplt
import numpy as np

import os
import matplotlib.pyplot as plt
from sklearn import preprocessing

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, roc_auc_score, roc_curve
from scipy import stats  # For in-built method to get PCC
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve

def datasetpreparation(datapath):
    idpdata_df = pd.read_csv(datapath)
    # idpdata_df=idpdata_df[['TIMESTAMP','M_ZONE_START', 'M_ZONE_FLAG', 'M_TRACK_TEMPERATURE', 'M_AIR_TEMPERATURE', 'M_NUM_WEATHER_FORECAST_SAMPLES', 'M_SESSION_TIME_LEFT', 'M_WEATHER_FORECAST_SAMPLES_M_SESSION_TYPE', 'M_TIME_OFFSET', 'M_WEATHER_FORECAST_SAMPLES_M_WEATHER', 'M_WEATHER_FORECAST_SAMPLES_M_TRACK_TEMPERATURE', 'M_TRACK_TEMPERATURE_CHANGE', 'M_WEATHER_FORECAST_SAMPLES_M_AIR_TEMPERATURE', 'M_AIR_TEMPERATURE_CHANGE', 'M_RAIN_PERCENTAGE', 'M_WEATHER']]

    idpdata_df.dtypes

    columnnames = list(idpdata_df.columns)
    print(columnnames)

    idpdata_df

    """#Data Preprocessing


    *  Missing Value Treatment
    *  Treating Outlier



    """

    idpdata_df.describe()

    """## Missing Value Treatment"""

    idpdata_df.isna().sum()

    # Plot amount of missingness
    msno.bar(idpdata_df)  # you can see pandas-profilin count part

    # plt.show()

    ### Forward Fill
    # Impute data DataFrame with ffill and bfill method
    idpdata_df_missingvaluetreatment = idpdata_df.fillna(0)

    idpdata_df_missingvaluetreatment.replace('NA', 0)

    # Plot amount of missingness
    msno.bar(idpdata_df_missingvaluetreatment)  # you can see pandas-profilin count part

    # plt.show()

    idpdata_df_missingvaluetreatment.isna().sum()

    idpdata_df = idpdata_df_missingvaluetreatment
    idpdata_df.head()

    idpdata_df.info()

    idpdata_df_missingvaluetreatment['NFI_PiN_2017'].replace('NA', 0)

    pd.to_numeric(idpdata_df_missingvaluetreatment['NFI_PiN_2017'])

    columns1 = ['Admin1', 'Admin2', 'Admin3', 'Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'CCCM_PiN_2017',
                'CCCM_Severity_2017']
    columns2 = ['Admin1', 'Admin2', 'Admin3', 'Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'Education_PiN_2017',
                'Education_Severity_2017']
    columns3 = ['Admin1', 'Admin2', 'Admin3', 'Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'Health_PiN_2017',
                'Health_Severity_2017']
    columns4 = ['Admin1', 'Admin2', 'Admin3', 'Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'Nutrition_PiN_2017',
                'Nutrition_Severity_2017']
    columns5 = ['Admin1', 'Admin2', 'Admin3', 'Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'Protection_PiN_2017',
                'Protection_Severity_2017']
    columns6 = ['Admin1', 'Admin2', 'Admin3', 'Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'NFI_PiN_2017',
                'NFI_Severity_2017']
    columns7 = ['Admin1', 'Admin2', 'Admin3', 'Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'Shelter_PiN_2017',
                'Shelter_Severity_2017']
    columns8 = ['Admin1', 'Admin2', 'Admin3', 'Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'WASH_PiN_2017',
                'WASH_Severity_2017']
    columns9 = ['Admin1', 'Admin2', 'Admin3', 'Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'FS_PiN_2017',
                'FS_Severity_2017']
    columns10 = ['Admin1', 'Admin2', 'Admin3', 'Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'ERL_PiN_2017',
                 'ERL_Severity_2017']
    columns11 = ['Admin1', 'Admin2', 'Admin3', 'Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'CCCM_PiN_2017',
                 'Education_PiN_2017', 'Health_PiN_2017', 'Nutrition_PiN_2017', 'Protection_PiN_2017', 'NFI_PiN_2017',
                 'Shelter_PiN_2017', 'WASH_PiN_2017', 'FS_PiN_2017', 'ERL_PiN_2017', 'Intersector_PiN']
    DatasetCCCM = idpdata_df[columns1]
    DatasetEducation = idpdata_df[columns2]
    DatasetHealth = idpdata_df[columns3]
    DatasetNutrition = idpdata_df[columns4]
    DatasetProtection = idpdata_df[columns5]
    DatasetNFI = idpdata_df[columns6]
    DatasetShelter = idpdata_df[columns7]
    DatasetWASH = idpdata_df[columns8]
    DatasetFS = idpdata_df[columns9]
    DatasetERL = idpdata_df[columns10]
    DatasetIntersector = idpdata_df[columns11]
    DatasetCCCM=DatasetCCCM.to_json()
    DatasetEducation=DatasetEducation.to_json()
    DatasetHealth=DatasetHealth.to_json()
    DatasetNutrition= DatasetNutrition.to_json()
    DatasetProtection=DatasetProtection.to_json()
    DatasetNFI=DatasetNFI.to_json()
    DatasetShelter=DatasetShelter.to_json()
    DatasetWASH=DatasetWASH.to_json()
    DatasetFS=DatasetFS.to_json()
    DatasetERL=DatasetERL.to_json()
    DatasetIntersector=DatasetIntersector.to_json()

    return DatasetCCCM,DatasetEducation,DatasetHealth,DatasetNutrition,DatasetProtection,DatasetNFI,DatasetShelter,DatasetWASH,DatasetFS,DatasetERL,DatasetIntersector



def datasetpreparationRAW(datapath):
    idpdata_df = pd.read_csv(datapath)
    # idpdata_df=idpdata_df[['TIMESTAMP','M_ZONE_START', 'M_ZONE_FLAG', 'M_TRACK_TEMPERATURE', 'M_AIR_TEMPERATURE', 'M_NUM_WEATHER_FORECAST_SAMPLES', 'M_SESSION_TIME_LEFT', 'M_WEATHER_FORECAST_SAMPLES_M_SESSION_TYPE', 'M_TIME_OFFSET', 'M_WEATHER_FORECAST_SAMPLES_M_WEATHER', 'M_WEATHER_FORECAST_SAMPLES_M_TRACK_TEMPERATURE', 'M_TRACK_TEMPERATURE_CHANGE', 'M_WEATHER_FORECAST_SAMPLES_M_AIR_TEMPERATURE', 'M_AIR_TEMPERATURE_CHANGE', 'M_RAIN_PERCENTAGE', 'M_WEATHER']]

    idpdata_df.dtypes

    columnnames = list(idpdata_df.columns)
    print(columnnames)

    idpdata_df

    """#Data Preprocessing


    *  Missing Value Treatment
    *  Treating Outlier



    """

    idpdata_df.describe()

    """## Missing Value Treatment"""

    idpdata_df.isna().sum()

    # Plot amount of missingness
    msno.bar(idpdata_df)  # you can see pandas-profilin count part

    # plt.show()

    ### Forward Fill
    # Impute data DataFrame with ffill and bfill method
    idpdata_df_missingvaluetreatment = idpdata_df.fillna(0)

    idpdata_df_missingvaluetreatment.replace('NA', 0)

    # Plot amount of missingness
    msno.bar(idpdata_df_missingvaluetreatment)  # you can see pandas-profilin count part

    # plt.show()

    idpdata_df_missingvaluetreatment.isna().sum()

    idpdata_df = idpdata_df_missingvaluetreatment
    idpdata_df.head()

    idpdata_df.info()

    idpdata_df_missingvaluetreatment['NFI_PiN_2017'].replace('NA', 0)

    pd.to_numeric(idpdata_df_missingvaluetreatment['NFI_PiN_2017'])

    columns1 = ['Admin1', 'Admin2', 'Admin3', 'Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'CCCM_PiN_2017',
                'CCCM_Severity_2017']
    columns2 = ['Admin1', 'Admin2', 'Admin3', 'Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'Education_PiN_2017',
                'Education_Severity_2017']
    columns3 = ['Admin1', 'Admin2', 'Admin3', 'Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'Health_PiN_2017',
                'Health_Severity_2017']
    columns4 = ['Admin1', 'Admin2', 'Admin3', 'Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'Nutrition_PiN_2017',
                'Nutrition_Severity_2017']
    columns5 = ['Admin1', 'Admin2', 'Admin3', 'Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'Protection_PiN_2017',
                'Protection_Severity_2017']
    columns6 = ['Admin1', 'Admin2', 'Admin3', 'Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'NFI_PiN_2017',
                'NFI_Severity_2017']
    columns7 = ['Admin1', 'Admin2', 'Admin3', 'Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'Shelter_PiN_2017',
                'Shelter_Severity_2017']
    columns8 = ['Admin1', 'Admin2', 'Admin3', 'Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'WASH_PiN_2017',
                'WASH_Severity_2017']
    columns9 = ['Admin1', 'Admin2', 'Admin3', 'Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'FS_PiN_2017',
                'FS_Severity_2017']
    columns10 = ['Admin1', 'Admin2', 'Admin3', 'Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'ERL_PiN_2017',
                 'ERL_Severity_2017']
    columns11 = ['Admin1', 'Admin2', 'Admin3', 'Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'CCCM_PiN_2017',
                 'Education_PiN_2017', 'Health_PiN_2017', 'Nutrition_PiN_2017', 'Protection_PiN_2017', 'NFI_PiN_2017',
                 'Shelter_PiN_2017', 'WASH_PiN_2017', 'FS_PiN_2017', 'ERL_PiN_2017', 'Intersector_PiN']
    idpdata_df.to_csv('cleanedidpdataset.csv')
    DatasetCCCM = idpdata_df[columns1]
    DatasetEducation = idpdata_df[columns2]
    DatasetHealth = idpdata_df[columns3]
    DatasetNutrition = idpdata_df[columns4]
    DatasetProtection = idpdata_df[columns5]
    DatasetNFI = idpdata_df[columns6]
    DatasetShelter = idpdata_df[columns7]
    DatasetWASH = idpdata_df[columns8]
    DatasetFS = idpdata_df[columns9]
    DatasetERL = idpdata_df[columns10]
    DatasetIntersector = idpdata_df[columns11]


    return DatasetCCCM,DatasetEducation,DatasetHealth,DatasetNutrition,DatasetProtection,DatasetNFI,DatasetShelter,DatasetWASH,DatasetFS,DatasetERL,DatasetIntersector

def results(target_y, predicted_y, modelname):
    y_valid = target_y
    preds = predicted_y
    model_name = modelname
    rms = np.sqrt(np.mean(np.power((np.array(y_valid) - np.array(preds)), 2)))
    print(model_name, ":rms", rms)
    score = r2_score(y_valid, preds)
    print(model_name, ":score", score)
    mae = mean_absolute_error(y_valid, preds)
    print(model_name, ":mae", mae)
    mse = mean_squared_error(y_valid, preds)
    print(model_name, ":mse", mse)
    pearson_coef, p_value = stats.pearsonr(y_valid, preds)
    print(model_name, ":pearson_coef", pearson_coef)
    print(model_name, ":p_value ", p_value)
    # Plot Histogram

    sns.distplot(preds, fit=norm);
    # Get the fitted parameters used by the function
    (mu1, sigma1) = norm.fit(preds)
    print(model_name, '\n Predicted mu = {:.2f} and sigma = {:.2f}\n'.format(mu1, sigma1))
    plt.legend(['Predicted Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu1, sigma1)],
               loc='best')
    plt.ylabel('Frequency')
    plt.title('{m} Predicted SalePrice distribution'.format(m=model_name))
    # Plot Histogram
    sns.distplot(y_valid, fit=norm);
    sns.distplot(preds, fit=norm);
    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(y_valid)
    (mu1, sigma1) = norm.fit(preds)
    print(model_name, '\n Valid mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    print(model_name, '\n Predicted mu = {:.2f} and sigma = {:.2f}\n'.format(mu1, sigma1))
    plt.legend(['Valid Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.legend(['Predicted Normal dist. ($\mu1=$ {:.2f} and $\sigma1=$ {:.2f} )'.format(mu1, sigma1)], loc='best')
    plt.ylabel('Frequency')
    plt.title('{m} Valid vs Predicted SalePrice distribution'.format(m=model_name))

    fig = plt.figure()
    res = stats.probplot(y_valid, plot=plt)
    res = stats.probplot(preds, plot=plt)
    plt.title('{m} Valid vs Predicted Probability Plot'.format(m=model_name))
    # plt.show()


def classfication_results(target_labels, predicted_labels, modelname):
    # print(classification_report(target_labels, predicted_labels))
    y_test = target_labels
    preds = predicted_labels
    rms = np.sqrt(np.mean(np.power((np.array(y_test) - np.array(preds)), 2)))
    score = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    # rms = math.sqrt(mse)
    pearson_coef, p_value = stats.pearsonr(y_test, preds)
    print("=======================================================================\n\n")
    print("Model Name: " + modelname)
    print("root mean square:", rms)
    print("score:", score)
    print("mean absolute error:", mae)
    print("mean squared error:", mse)
    print("pearson_coef:", pearson_coef)
    print("p_value:", p_value)
    print("=======================================================================\n\n")


"""## Linear regression model"""


def linear_regression_model(dataset):
    train_x, test_x, train_y, test_y = train_test_split(dataset.values[:, 3:-1], dataset.values[:, -1])
    linear_regression = linear_model.LinearRegression()
    linear_regression.fit(train_x, train_y)
    predicted_y = linear_regression.predict(test_x)
    print('Coefficients: \n', linear_regression.coef_)
    print("\nMean squared error: ", mean_squared_error(test_y, predicted_y))
    print('Variance score: %.2f' % r2_score(test_y, predicted_y))
    mse = mean_squared_error(test_y, predicted_y)
    variance_score = r2_score(test_y, predicted_y)
    #results(test_y, predicted_y, "Linear Regression")
    classfication_results(test_y, predicted_y, "Linear regression")
    return mse, variance_score,predicted_y


"""## SVM model"""


def support_vector_machine_model(dataset):
    train_x, test_x, train_y, test_y = train_test_split(dataset.values[:, 3:-1],dataset.values[:, -1])
    svm_model = svm.SVR()
    svm_model.fit(train_x, train_y)
    predicted_y = svm_model.predict(test_x)
    print("Mean squared error: ", mean_squared_error(test_y, predicted_y))
    print('Variance score: %.2f' % r2_score(test_y, predicted_y))
    mse = mean_squared_error(test_y, predicted_y)
    variance_score = r2_score(test_y, predicted_y)
    #results(test_y, predicted_y, "SVM")
    classfication_results(test_y, predicted_y, "SVM")
    return mse, variance_score,predicted_y


"""## Random forest model"""


def random_forest_model(dataset):
    train_x, test_x, train_y, test_y = train_test_split(dataset.values[:, 3:-1], dataset.values[:, -1])
    random_forest = RandomForestRegressor()
    random_forest.fit(train_x, train_y)
    predicted_y = random_forest.predict(test_x)

    print("Mean squared error: ", mean_squared_error(test_y, predicted_y))
    print('Variance score: %.2f' % r2_score(test_y, predicted_y))

    mse = mean_squared_error(test_y, predicted_y)
    variance_score = r2_score(test_y, predicted_y)
    #results(test_y, predicted_y, "Random Forest")
    classfication_results(test_y, predicted_y, "Random Forest")
    return mse, variance_score,predicted_y


"""# Performance Comparision functions

### Function for generating the graph
"""


def generate_plot(title, ticks, dataset, color_number):
    colors = ["slateblue", "mediumseagreen", "tomato"]
    plt.figure(figsize=(8, 6))

    ax = plt.subplot()
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.xticks(np.arange(len(ticks)), ticks, fontsize=10, rotation=30)
    plt.title(title, fontsize=22)
    plt.bar(ticks, dataset, linewidth=1.2, color=colors[color_number])


def comparisonofmodels(mse_values, variance_score):
    ticks = ["Linear Regression", "SVM", "Random Forest"]
    generate_plot("Plot of MSE values", ticks, mse_values, 1)
    generate_plot("Plot of Variance scores", ticks, variance_score, 1)


"""# Model Implementation and Performance Analysis"""

def severityanalysis(dataset):
  #dataset= pd.read_csv(datasetpath)
  mse_values1, variance_score1,predicted_y1 = linear_regression_model(dataset)
  mse_values2, variance_score2,predicted_y2 = support_vector_machine_model(dataset)
  mse_values3, variance_score3,predicted_y3  = random_forest_model(dataset)
  mse_values = [mse_values1, mse_values2, mse_values3]
  variance_score = [variance_score1, variance_score2, variance_score3]
  predicted_y=[predicted_y1, predicted_y2, predicted_y3]
  mse_values= pd.DataFrame(mse_values)
  variance_score = pd.DataFrame(variance_score)
  predicted_y = pd.DataFrame(predicted_y)
  mse_values.to_csv('mse_values.csv')
  variance_score.to_csv('variance_score.csv')
  predicted_y.to_csv('predicted_y.csv')
  predicted_y1= pd.DataFrame(predicted_y1)
  predicted_y2 = pd.DataFrame(predicted_y2)
  predicted_y3 = pd.DataFrame(predicted_y2)
  results_final = [
      {'LR': predicted_y1.to_json(),
      'SVM':predicted_y2.to_json(),
      'RF': predicted_y3.to_json()
      }]
  return results_final

if __name__ == "__main__":
    DatasetCCCM, DatasetEducation, DatasetHealth, DatasetNutrition, DatasetProtection, DatasetNFI, DatasetShelter, DatasetWASH, DatasetFS, DatasetERL, DatasetIntersector=datasetpreparationRAW('datasetsyria2.csv')
    results_final=severityanalysis(DatasetCCCM)
    print(results_final)

