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

# import sweetviz as sv

"""#Data Acquistion"""

idpdata_df = pd.read_csv('datasetsyria2.csv')
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

columnnames = list(idpdata_df.columns)
print(columnnames)

"""#Exploratory data analysis Analysis

## Investigating Outliers
"""

columnnames = list(idpdata_df.columns)
print(columnnames[15:])

for columnname in columnnames[15:]:
    plt.figure(figsize=(22, 10))
    sns.distplot(idpdata_df[columnname], fit=norm);

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(idpdata_df[columnname])
    print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
               loc='best')
    plt.ylabel('Frequency')
    plt.title(columnname + ' distribution ')

    fig = plt.figure()
    res = stats.probplot(idpdata_df[columnname], plot=plt)
    # plt.show()
    print("Skewness: %f" % idpdata_df[columnname].skew())
    print("Kurtosis: %f" % idpdata_df[columnname].kurt())

for columnname in columnnames[3:]:
    plt.figure(figsize=(22, 10))
    idpdata_df[columnname].plot()
    plt.title(columnname)
    # plt.show()

idpdata_df.plot()
# plt.show()

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
DatasetCCCM.to_csv('DatasetCCCM.csv')
DatasetEducation = idpdata_df[columns2]
DatasetEducation.to_csv('DatasetEducation.csv')
DatasetHealth = idpdata_df[columns3]
DatasetHealth.to_csv('DatasetHealth.csv')
DatasetNutrition = idpdata_df[columns4]
DatasetNutrition.to_csv('DatasetNutrition.csv')
DatasetProtection = idpdata_df[columns5]
DatasetProtection.to_csv('DatasetProtection.csv')
DatasetNFI = idpdata_df[columns6]
DatasetNFI.to_csv('DatasetNFI.csv')
DatasetShelter = idpdata_df[columns7]
DatasetShelter.to_csv('DatasetShelter.csv')
DatasetWASH = idpdata_df[columns8]
DatasetWASH.to_csv('DatasetWASH.csv')
DatasetFS = idpdata_df[columns9]
DatasetFS.to_csv('DatasetFS.csv')
DatasetERL = idpdata_df[columns10]
DatasetERL.to_csv('DatasetERL.csv')
DatasetIntersector = idpdata_df[columns11]
DatasetIntersector.to_csv('DatasetIntersector.csv')

configcolumns1 = ['Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'CCCM_PiN_2017', 'CCCM_Severity_2017']
configcolumns2 = ['Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'Education_PiN_2017', 'Education_Severity_2017']
configcolumns3 = ['Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'Health_PiN_2017', 'Health_Severity_2017']
configcolumns4 = ['Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'Nutrition_PiN_2017', 'Nutrition_Severity_2017']
configcolumns5 = ['Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'Protection_PiN_2017', 'Protection_Severity_2017']
configcolumns6 = ['Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'NFI_PiN_2017', 'NFI_Severity_2017']
configcolumns7 = ['Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'Shelter_PiN_2017', 'Shelter_Severity_2017']
configcolumns8 = ['Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'WASH_PiN_2017', 'WASH_Severity_2017']
configcolumns9 = ['Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'FS_PiN_2017', 'FS_Severity_2017']
configcolumns10 = ['Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'ERL_PiN_2017', 'ERL_Severity_2017']
configcolumns11 = ['Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'CCCM_PiN_2017', 'Education_PiN_2017',
                   'Health_PiN_2017', 'Nutrition_PiN_2017', 'Protection_PiN_2017', 'NFI_PiN_2017', 'Shelter_PiN_2017',
                   'WASH_PiN_2017', 'FS_PiN_2017', 'ERL_PiN_2017', 'Intersector_PiN']

columnnames = list(idpdata_df.columns)
print(columnnames)

for columnname in columnnames[3:]:
    plt.figure(figsize=(22, 10))
    plt.bar(idpdata_df['Admin1'], idpdata_df[columnname])
    plt.title(columnname)
    # plt.show()

for columnname in columnnames[3:]:
    plt.figure(figsize=(22, 10))
    plt.bar(idpdata_df['Admin2'], idpdata_df[columnname])
    plt.title(columnname)
    # plt.show()

"""## Statistical data analysis

### CCCM Severity
"""

columnnames = configcolumns1
print(columnnames)

for columnname in columnnames:
    plt.figure(figsize=(22, 10))
    DatasetCCCM[columnname].plot()
    plt.title(columnname)
    # plt.show()

# Pair Grid
for columnname in columns1[3:-1]:
    plt.figure(figsize=(22, 10))
    sns.jointplot(x=DatasetCCCM[columnname], y=DatasetCCCM['CCCM_Severity_2017'], kind='reg')
    plt.title(columnname + " vs CCCM Severity")
    # plt.show()

"""### Education Severity"""

columnnames = configcolumns2
print(columnnames)

for columnname in columnnames:
    plt.figure(figsize=(22, 10))
    DatasetEducation[columnname].plot()
    plt.title(columnname)
    # plt.show()

# Pair Grid
for columnname in columns2[3:-1]:
    plt.figure(figsize=(22, 10))
    sns.jointplot(x=DatasetEducation[columnname], y=DatasetEducation['Education_Severity_2017'], kind='reg')
    plt.title(columnname + " vs Education_Severity")
    # plt.show()

"""### Health Severity"""

columnnames = configcolumns3
print(columnnames)

for columnname in columnnames:
    plt.figure(figsize=(22, 10))
    DatasetHealth[columnname].plot()
    plt.title(columnname)
    # plt.show()

# Pair Grid
for columnname in columns3[3:-1]:
    plt.figure(figsize=(22, 10))
    sns.jointplot(x=DatasetHealth[columnname], y=DatasetHealth['Health_Severity_2017'], kind='reg')
    plt.title(columnname + " vs Health_Severity")
    # plt.show()

"""### Nutrition Severity"""

columnnames = configcolumns4
print(columnnames)

for columnname in columnnames:
    plt.figure(figsize=(22, 10))
    DatasetNutrition[columnname].plot()
    plt.title(columnname)
    # plt.show()

# Pair Grid
for columnname in columns4[3:-1]:
    plt.figure(figsize=(22, 10))
    sns.jointplot(x=DatasetNutrition[columnname], y=DatasetNutrition['Nutrition_Severity_2017'], kind='reg')
    plt.title(columnname + " vs Nutrition_Severity")
    # plt.show()

"""### Protection Severity"""

columnnames = configcolumns5
print(columnnames)

for columnname in columnnames:
    plt.figure(figsize=(22, 10))
    DatasetProtection[columnname].plot()
    plt.title(columnname)
    # plt.show()

# Pair Grid
for columnname in columns5[3:-1]:
    plt.figure(figsize=(22, 10))
    sns.jointplot(x=DatasetProtection[columnname], y=DatasetProtection['Protection_Severity_2017'], kind='reg')
    plt.title(columnname + " vs Protection_Severity_2017")
    # plt.show()

"""feature_config1 = sv.FeatureConfig(force_num=configcolumns5)
report = sv.analyze(DatasetProtection, target_feat=columns5[-1], feat_cfg=feature_config1)
report.show_notebook()"""

"""### NFI Severity"""

columnnames = configcolumns6
print(columnnames)

for columnname in columnnames:
    plt.figure(figsize=(22, 10))
    DatasetNFI[columnname].plot()
    plt.title(columnname)
    # plt.show()

# Pair Grid
for columnname in columns6[3:-1]:
    plt.figure(figsize=(22, 10))
    sns.jointplot(x=DatasetNFI[columnname], y=DatasetNFI['NFI_Severity_2017'], kind='reg')
    plt.title(columnname + "vs NFI_Severity_2017")
    # plt.show()

"""### Shelter Severity"""

columnnames = configcolumns7
print(columnnames)

for columnname in columnnames:
    plt.figure(figsize=(22, 10))
    DatasetShelter[columnname].plot()
    plt.title(columnname)
    # plt.show()

# Pair Grid
for columnname in columns7[3:-1]:
    plt.figure(figsize=(22, 10))
    sns.jointplot(x=DatasetShelter[columnname], y=DatasetShelter['Shelter_Severity_2017'], kind='reg')
    plt.title(columnname + "vs 'Nutrition_Severity")
    # plt.show()

"""### WASH Severity"""

columnnames = configcolumns8
print(columnnames)

for columnname in columnnames:
    plt.figure(figsize=(22, 10))
    DatasetWASH[columnname].plot()
    plt.title(columnname)
    # plt.show()

    # Pair Grid
    for columnname in columns8[3:-1]:
        plt.figure(figsize=(22, 10))
        sns.jointplot(x=DatasetWASH[columnname], y=DatasetWASH['WASH_Severity_2017'], kind='reg')
        plt.title(columnname + " vs WASH_Severity")
        # plt.show()

"""### FS Severity"""

columnnames = configcolumns9
print(columnnames)

for columnname in columnnames:
    plt.figure(figsize=(22, 10))
    DatasetFS[columnname].plot()
    plt.title(columnname)
    # plt.show()

# Pair Grid
for columnname in columns9[3:-1]:
    plt.figure(figsize=(22, 10))
    sns.jointplot(x=DatasetFS[columnname], y=DatasetFS['FS_Severity_2017'], kind='reg')
    plt.title(columnname + " vs FS_Severity")
    # plt.show()

"""### ERL Severity"""

columnnames = configcolumns10
print(columnnames)

for columnname in columnnames:
    plt.figure(figsize=(22, 10))
    DatasetERL[columnname].plot()
    plt.title(columnname)
    # plt.show()

# Pair Grid
for columnname in columns10[3:-1]:
    plt.figure(figsize=(22, 10))
    sns.jointplot(x=DatasetERL[columnname], y=DatasetERL['ERL_Severity_2017'], kind='reg')
    plt.title(columnname + " vs ERL_Severity")
    # plt.show()

"""### Intersector_PiN"""

columnnames = configcolumns11
print(columnnames)

for columnname in columnnames:
    plt.figure(figsize=(22, 10))
    DatasetIntersector[columnname].plot()
    plt.title(columnname)
    # plt.show()

# Pair Grid
for columnname in configcolumns11[3:-1]:
    plt.figure(figsize=(22, 10))
    sns.jointplot(x=DatasetIntersector[columnname], y=DatasetIntersector['Intersector_PiN'], kind='reg')
    plt.title(columnname + "vs 'Intersector_PiN")
    # plt.show()

"""#Variables Selection
correlation matrix and heatmap
you can select  the Most correlated variables

## Over All correlation
"""

# Correlation Matrix 
corrmat = idpdata_df.corr()
print("Correlation Matrix", corrmat)
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.title('Correlation Matrix ')
# plt.show()

"""## CCCM Severity"""

# Correlation Matrix 
corrmat = DatasetCCCM.corr()
print("Correlation Matrix", corrmat)
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.title('Correlation Matrix ')
# plt.show()

# Top 10 Heatmap
k = len(columns1)  # number of variables for heatmap
columnstarget = columns1[-1]
cols = corrmat.nlargest(k, columnstarget)[columnstarget].index
print("cols", cols)
cm = np.corrcoef(DatasetCCCM[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels=cols.values, xticklabels=cols.values)
plt.title('10 Variables Correlation Matrix')
# plt.show()
most_corr = pd.DataFrame(cols)
most_corr.columns = ['Most Correlated Features']
print(most_corr)
corrs = most_corr.values
most_corr_list = list(most_corr["Most Correlated Features"])

print(most_corr_list)

most_corr_list = list(most_corr["Most Correlated Features"])

print(most_corr_list)

"""## Education Severity"""

# Correlation Matrix 
corrmat = DatasetEducation.corr()
print("Correlation Matrix", corrmat)
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.title('Correlation Matrix ')
# plt.show()

# Top 10 Heatmap
k = len(columns2)  # number of variables for heatmap
columnstarget = columns2[-1]
cols = corrmat.nlargest(k, columnstarget)[columnstarget].index
print("cols", cols)
cm = np.corrcoef(DatasetEducation[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels=cols.values, xticklabels=cols.values)
plt.title('10 Variables Correlation Matrix')
# plt.show()
most_corr = pd.DataFrame(cols)
most_corr.columns = ['Most Correlated Features']
print(most_corr)
corrs = most_corr.values
most_corr_list = list(most_corr["Most Correlated Features"])

print(most_corr_list)

most_corr_list = list(most_corr["Most Correlated Features"])

print(most_corr_list)

"""## Health Severity"""

# Correlation Matrix 
corrmat = DatasetHealth.corr()
print("Correlation Matrix", corrmat)
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.title('Correlation Matrix ')
# plt.show()

columns3

# Top 10 Heatmap
k = len(columns3)  # number of variables for heatmap
columnstarget = columns3[-1]
cols = corrmat.nlargest(k, columnstarget)[columnstarget].index
print("cols", cols)
cm = np.corrcoef(DatasetHealth[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels=cols.values, xticklabels=cols.values)
plt.title('10 Variables Correlation Matrix')
# plt.show()
most_corr = pd.DataFrame(cols)
most_corr.columns = ['Most Correlated Features']
print(most_corr)
corrs = most_corr.values
most_corr_list = list(most_corr["Most Correlated Features"])

print(most_corr_list)

most_corr_list = list(most_corr["Most Correlated Features"])

print(most_corr_list)

"""## Nutrition Severity"""

# Correlation Matrix 
corrmat = DatasetNutrition.corr()
print("Correlation Matrix", corrmat)
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.title('Correlation Matrix ')
# plt.show()

# Top 10 Heatmap
k = len(columns4)  # number of variables for heatmap
columnstarget = columns4[-1]
cols = corrmat.nlargest(k, columnstarget)[columnstarget].index
print("cols", cols)
cm = np.corrcoef(DatasetNutrition[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels=cols.values, xticklabels=cols.values)
plt.title('10 Variables Correlation Matrix')
# plt.show()
most_corr = pd.DataFrame(cols)
most_corr.columns = ['Most Correlated Features']
print(most_corr)
corrs = most_corr.values
most_corr_list = list(most_corr["Most Correlated Features"])

print(most_corr_list)

most_corr_list = list(most_corr["Most Correlated Features"])

print(most_corr_list)

"""## Protection Severity"""

# Correlation Matrix 
corrmat = DatasetProtection.corr()
print("Correlation Matrix", corrmat)
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.title('Correlation Matrix ')
# plt.show()

# Top 10 Heatmap
k = len(columns5)  # number of variables for heatmap
columnstarget = columns5[-1]
cols = corrmat.nlargest(k, columnstarget)[columnstarget].index
print("cols", cols)
cm = np.corrcoef(DatasetProtection[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels=cols.values, xticklabels=cols.values)
plt.title('10 Variables Correlation Matrix')
# plt.show()
most_corr = pd.DataFrame(cols)
most_corr.columns = ['Most Correlated Features']
print(most_corr)
corrs = most_corr.values
most_corr_list = list(most_corr["Most Correlated Features"])

print(most_corr_list)

most_corr_list = list(most_corr["Most Correlated Features"])

print(most_corr_list)

"""## NFI Severity"""

# Correlation Matrix 
corrmat = DatasetNFI.corr()
print("Correlation Matrix", corrmat)
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.title('Correlation Matrix ')
# plt.show()

# Top 10 Heatmap
k = len(columns6)  # number of variables for heatmap
columnstarget = columns6[-1]
cols = corrmat.nlargest(k, columnstarget)[columnstarget].index
print("cols", cols)
cm = np.corrcoef(DatasetNFI[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels=cols.values, xticklabels=cols.values)
plt.title('10 Variables Correlation Matrix')
# plt.show()
most_corr = pd.DataFrame(cols)
most_corr.columns = ['Most Correlated Features']
print(most_corr)
corrs = most_corr.values
most_corr_list = list(most_corr["Most Correlated Features"])

print(most_corr_list)

most_corr_list = list(most_corr["Most Correlated Features"])

print(most_corr_list)

"""## Shelter Severity"""

# Correlation Matrix 
corrmat = DatasetShelter.corr()
print("Correlation Matrix", corrmat)
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.title('Correlation Matrix ')
# plt.show()

# Top 10 Heatmap
k = len(columns7)  # number of variables for heatmap
columnstarget = columns7[-1]
cols = corrmat.nlargest(k, columnstarget)[columnstarget].index
print("cols", cols)
cm = np.corrcoef(DatasetShelter[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels=cols.values, xticklabels=cols.values)
plt.title('10 Variables Correlation Matrix')
# plt.show()
most_corr = pd.DataFrame(cols)
most_corr.columns = ['Most Correlated Features']
print(most_corr)
corrs = most_corr.values
most_corr_list = list(most_corr["Most Correlated Features"])

print(most_corr_list)

most_corr_list = list(most_corr["Most Correlated Features"])

print(most_corr_list)

"""## WASH Severity"""

# Correlation Matrix 
corrmat = DatasetWASH.corr()
print("Correlation Matrix", corrmat)
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.title('Correlation Matrix ')
# plt.show()

# Top 10 Heatmap
k = len(columns8)  # number of variables for heatmap
columnstarget = columns8[-1]
cols = corrmat.nlargest(k, columnstarget)[columnstarget].index
print("cols", cols)
cm = np.corrcoef(DatasetWASH[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels=cols.values, xticklabels=cols.values)
plt.title('10 Variables Correlation Matrix')
# plt.show()
most_corr = pd.DataFrame(cols)
most_corr.columns = ['Most Correlated Features']
print(most_corr)
corrs = most_corr.values
most_corr_list = list(most_corr["Most Correlated Features"])

print(most_corr_list)

most_corr_list = list(most_corr["Most Correlated Features"])

print(most_corr_list)

"""## FS Severity"""

# Correlation Matrix 
corrmat = DatasetFS.corr()
print("Correlation Matrix", corrmat)
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.title('Correlation Matrix ')
# plt.show()

# Top 10 Heatmap
k = len(columns9)  # number of variables for heatmap
columnstarget = columns9[-1]
cols = corrmat.nlargest(k, columnstarget)[columnstarget].index
print("cols", cols)
cm = np.corrcoef(DatasetFS[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels=cols.values, xticklabels=cols.values)
plt.title('10 Variables Correlation Matrix')
# plt.show()
most_corr = pd.DataFrame(cols)
most_corr.columns = ['Most Correlated Features']
print(most_corr)
corrs = most_corr.values
most_corr_list = list(most_corr["Most Correlated Features"])

print(most_corr_list)

most_corr_list = list(most_corr["Most Correlated Features"])

print(most_corr_list)

"""## ERL Severity"""

# Correlation Matrix 
corrmat = DatasetERL.corr()
print("Correlation Matrix", corrmat)
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.title('Correlation Matrix ')
# plt.show()

# Top 10 Heatmap
k = len(columns10)  # number of variables for heatmap
columnstarget = columns10[-1]
cols = corrmat.nlargest(k, columnstarget)[columnstarget].index
print("cols", cols)
cm = np.corrcoef(DatasetERL[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels=cols.values, xticklabels=cols.values)
plt.title('10 Variables Correlation Matrix')
# plt.show()
most_corr = pd.DataFrame(cols)
most_corr.columns = ['Most Correlated Features']
print(most_corr)
corrs = most_corr.values
most_corr_list = list(most_corr["Most Correlated Features"])

print(most_corr_list)

most_corr_list = list(most_corr["Most Correlated Features"])

print(most_corr_list)

"""## Intersector Pin"""

# Correlation Matrix 
corrmat = DatasetIntersector.corr()
print("Correlation Matrix", corrmat)
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.title('Correlation Matrix ')
# plt.show()

# Top 10 Heatmap
k = len(columns11)  # number of variables for heatmap
columnstarget = columns11[-1]
cols = corrmat.nlargest(k, columnstarget)[columnstarget].index
print("cols", cols)
cm = np.corrcoef(DatasetIntersector[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels=cols.values, xticklabels=cols.values)
plt.title('10 Variables Correlation Matrix')
# plt.show()
most_corr = pd.DataFrame(cols)
most_corr.columns = ['Most Correlated Features']
print(most_corr)
corrs = most_corr.values
most_corr_list = list(most_corr["Most Correlated Features"])

print(most_corr_list)

most_corr_list = list(most_corr["Most Correlated Features"])

print(most_corr_list)

"""#Data Transformation



*   Convert timestemps and dataset into time series
*   Split data into traing and testing with ratio of 0.2


"""

columns1 = ['Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'CCCM_PiN_2017', 'CCCM_Severity_2017']
columns2 = ['Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'Education_PiN_2017', 'Education_Severity_2017']
columns3 = ['Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'Health_PiN_2017', 'Health_Severity_2017']
columns4 = ['Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'Nutrition_PiN_2017', 'Nutrition_Severity_2017']
columns5 = ['Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'Protection_PiN_2017', 'Protection_Severity_2017']
columns6 = ['Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'NFI_PiN_2017', 'NFI_Severity_2017']
columns7 = ['Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'Shelter_PiN_2017', 'Shelter_Severity_2017']
columns8 = ['Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'WASH_PiN_2017', 'WASH_Severity_2017']
columns9 = ['Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'FS_PiN_2017', 'FS_Severity_2017']
columns10 = ['Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'ERL_PiN_2017', 'ERL_Severity_2017']
columns11 = ['Pop_HNO _2017', 'IDPs_estimates_for_the_HNO_2017', 'CCCM_PiN_2017', 'Education_PiN_2017',
             'Health_PiN_2017', 'Nutrition_PiN_2017', 'Protection_PiN_2017', 'NFI_PiN_2017', 'Shelter_PiN_2017',
             'WASH_PiN_2017', 'FS_PiN_2017', 'ERL_PiN_2017', 'Intersector_PiN']

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

CCCMfeature_train, CCCMfeature_test, CCCMtarget_train, CCCMtarget_test = train_test_split(DatasetCCCM.values[:, :-1],
                                                                                          DatasetCCCM.values[:, -1])

Educationfeature_train, Educationfeature_test, Educationtarget_train, Educationtarget_test = train_test_split(
    DatasetEducation.values[:, :-1], DatasetEducation.values[:, -1])

Healthfeature_train, Healthfeature_test, Healthtarget_train, Healthtarget_test = train_test_split(
    DatasetHealth.values[:, :-1], DatasetHealth.values[:, -1])

Nutritionfeature_train, Nutritionfeature_test, Nutritiontarget_train, Nutritiontarget_test = train_test_split(
    DatasetNutrition.values[:, :-1], DatasetNutrition.values[:, -1])

Protectionfeature_train, Protectionfeature_test, Protectiontarget_train, Protectiontarget_test = train_test_split(
    DatasetProtection.values[:, :-1], DatasetProtection.values[:, -1])

NFIfeature_train, NFIfeature_test, NFItarget_train, NFItarget_test = train_test_split(DatasetNFI.values[:, :-1],
                                                                                      DatasetNFI.values[:, -1])

Shelterfeature_train, Shelterfeature_test, Sheltertarget_train, Sheltertarget_test = train_test_split(
    DatasetShelter.values[:, :-1], DatasetShelter.values[:, -1])

WASHfeature_train, WASHfeature_test, WASHtarget_train, WASHtarget_test = train_test_split(DatasetWASH.values[:, :-1],
                                                                                          DatasetWASH.values[:, -1])

FSfeature_train, FSfeature_test, FStarget_train, FStarget_test = train_test_split(DatasetFS.values[:, :-1],
                                                                                  DatasetFS.values[:, -1])

ERLfeature_train, ERLfeature_test, ERLtarget_train, ERLtarget_test = train_test_split(DatasetERL.values[:, :-1],
                                                                                      DatasetERL.values[:, -1])

Intersectorfeature_train, Intersectorfeature_test, Intersectortarget_train, Intersectortarget_test = train_test_split(
    DatasetIntersector.values[:, :-1], DatasetIntersector.values[:, -1])

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


def linear_regression_model(train_x, test_x, train_y, test_y):
    linear_regression = linear_model.LinearRegression()
    linear_regression.fit(train_x, train_y)
    predicted_y = linear_regression.predict(test_x)
    print('Coefficients: \n', linear_regression.coef_)
    print("\nMean squared error: ", mean_squared_error(test_y, predicted_y))
    print('Variance score: %.2f' % r2_score(test_y, predicted_y))
    mse = mean_squared_error(test_y, predicted_y)
    variance_score = r2_score(test_y, predicted_y)
    results(test_y, predicted_y, "Linear Regression")
    return mse, variance_score,predicted_y


"""## SVM model"""


def support_vector_machine_model(dataset):
    train_x, test_x, train_y, test_y = train_test_split(dataset.values[:, :-1],dataset.values[:, -1])
    svm_model = svm.SVR()
    svm_model.fit(train_x, train_y)
    predicted_y = svm_model.predict(test_x)
    print("Mean squared error: ", mean_squared_error(test_y, predicted_y))
    print('Variance score: %.2f' % r2_score(test_y, predicted_y))
    mse = mean_squared_error(test_y, predicted_y)
    variance_score = r2_score(test_y, predicted_y)
    results(test_y, predicted_y, "SVM")
    classfication_results(test_y, predicted_y, "SVM")
    return mse, variance_score,predicted_y


"""## Random forest model"""


def random_forest_model(train_x, test_x, train_y, test_y):
    random_forest = RandomForestRegressor()
    random_forest.fit(train_x, train_y)
    predicted_y = random_forest.predict(test_x)

    print("Mean squared error: ", mean_squared_error(test_y, predicted_y))
    print('Variance score: %.2f' % r2_score(test_y, predicted_y))

    mse = mean_squared_error(test_y, predicted_y)
    variance_score = r2_score(test_y, predicted_y)
    results(test_y, predicted_y, "Random Forest")
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
  mse_values1, variance_score1,predicted_y1 = linear_regression_model(dataset)
  mse_values2, variance_score2,predicted_y2 = support_vector_machine_model(dataset)
  mse_values3, variance_score3,predicted_y3  = random_forest_model(dataset)
  mse_values = [mse_values1, mse_values2, mse_values3]
  variance_score = [variance_score1, variance_score2, variance_score3]
  predicted_y=[predicted_y1, predicted_y2, predicted_y3]
  comparisonofmodels(mse_values, variance_score)
  return predicted_y
