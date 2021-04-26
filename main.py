import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer as Imputer

from KMeans import showDifferentParamsPerformance, showModelPerformanceKMeans
from Hierarchical_Clustering import showModelPerformanceHierarchical
from DBSCAN import showModelPerformanceDBSCAN


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



data = pd.read_csv('weatherAUS.csv')

# Impression des informations sur le dataset
print(data.head())
print(data.info())
print(data.describe())


# Compte du nombre de valeurs pour chacune des classes
print("Values per class: \n", data['RainTomorrow'].value_counts())


# Compte du nombre de valeurs manquantes
print("Missing values per attribute: \n", data.isnull().sum())


# Remplacement des dates par le jour de l'année
data.loc[:, "Date"] = data.loc[:, "Date"].apply(lambda x: pd.Period(x, freq='H').dayofyear)

# Remplissage des valeurs manquantes de la façon approprié
fillWithMean = ["MinTemp", "MaxTemp", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Temp9am", "Temp3pm"]

fillWithZero = ["Evaporation", "Sunshine", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Cloud9am", "Cloud3pm",
                "Rainfall"]

fillWithMostFrequent = ["RainToday", "RainTomorrow"]

fillWithNoWind = ["WindGustDir", "WindDir9am", "WindDir3pm"]

imputer = Imputer(strategy='mean')
data.loc[:, fillWithMean] = imputer.fit_transform(data[fillWithMean])

imputer = Imputer(strategy='most_frequent')
data.loc[:, fillWithMostFrequent] = imputer.fit_transform(data[fillWithMostFrequent])

data.loc[:, fillWithZero] = data.loc[:, fillWithZero].fillna(value=0, axis=1)

data.loc[:, fillWithNoWind] = data.loc[:, fillWithNoWind].fillna(value="No Wind", axis=1)

# Encodage des attributs qui sont des classes

labelEncoder = LabelEncoder()
data["RainToday"] = labelEncoder.fit_transform(data["RainToday"])
data["RainTomorrow"] = labelEncoder.fit_transform(data["RainTomorrow"])
data["Location"] = labelEncoder.fit_transform(data["Location"])
data["WindGustDir"] = labelEncoder.fit_transform(data["WindGustDir"])
data["WindDir9am"] = labelEncoder.fit_transform(data["WindDir9am"])
data["WindDir3pm"] = labelEncoder.fit_transform(data["WindDir3pm"])

# Séparation des données et des labels
X_train = data.loc[:, data.columns != 'RainTomorrow']

labels = data.RainTomorrow
y_train = pd.Series(data=labels, name="RainTomorrow")

# Mise à l'échelle des attributs
scaler = StandardScaler()
X_train.loc[:, :] = scaler.fit_transform(X_train)


# K-MEANS

# Pour trouver le meilleur nombre de clusters
showDifferentParamsPerformance(X_train, y_train, 1, 31)

# On trouve la precision de chaque cluster
showModelPerformanceKMeans(X_train, y_train, 23)


# Clustering hierarchique
showModelPerformanceHierarchical(X_train, y_train)


# DBSCAN
showModelPerformanceDBSCAN(X_train, y_train, 1.5, 5)

