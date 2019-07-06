# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 00:07:07 2019

@author: Ullas
Following the code from the book - Hands on Machine Learning with Scikit 
Learn and TensorFlow - Chapter 2:End-to-End Machine Learning Project 
"""

import pandas as pd

#Reading the CSV file and displaying the default 5 rows
housing = pd.read_csv("E://Python//bookCode//housing.csv")
housing.head()

#To know the type of data along with the count 
housing.info()

#specifically to know how many values correspond to each category in the column
housing["ocean_proximity"].value_counts()

#Gives statistical information such as min max count and quartiles
housing.describe()

import matplotlib.pyplot as plt
#to generate histograms for the data
housing.hist(bins=50, figsize=(20,15))
plt.show()

#splitting data into training and test data 
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state =42)

housing["median_income"].hist()
import numpy as np
#Creating a new column
housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
#Everything below 5 will be as such and above 5 will be replaced with 5
housing["income_cat"].where(housing["income_cat"]<5, 5.0, inplace = True)
housing["income_cat"].hist()
plt.show()

#to represent the actual population stratified split is used
from sklearn.model_selection import StratifiedShuffleSplit
#representing the same proportion of category as in population
split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
strat_train_set.describe()
strat_test_set.describe()
strat_train_set.hist(bins=50, figsize=(12,7))
strat_test_set.hist(bins=50, figsize=(12,7))
plt.show()

#removing the new column
strat_train_set.drop("income_cat", axis=1, inplace=True)
strat_test_set.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()

#scatter plot 
housing.plot(kind="scatter", x="longitude", y="latitude")
plt.show()
#scatter plot with density
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

"""plotting population with the median house values. s option defines the 
radius. Hence dividing the population by 100. color is represented using 
c option and is set to the value of median house values. It uses predefinded
color map, jet, that ranges from blue to red."""
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
s=housing["population"]/100, label="population",
c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()

#For correlation
corr_matrix = housing.corr()
corr_matrix
corr_matrix["median_house_value"].sort_values(ascending=False)

#for plotting correlation matrix 
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,7))
plt.show()

#for a pair we are plotting a correlation matrix graph 
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
plt.axis([0, 16, 0, 550000])

#combining few data 
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
housing.head(10)
#new correlation
corr_matrix = housing.corr()
corr_matrix
corr_matrix["median_income"].sort_values(ascending=False)

#Separating the target variable
housing = strat_train_set.drop("median_house_value",axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
#null check
housing.isnull().any()

#Only taking those rows for which there is null value present. 
sample_incomplete_rows = housing[housing.isnull().any(axis=1)]
sample_incomplete_rows
#either remove all rows that have na in total bedrooms. This will empty df
sample_incomplete_rows.dropna(subset=["total_bedrooms"])
#or drop the column 
sample_incomplete_rows.drop("total_bedrooms", axis=1) 
#or replace with the median 
median = housing["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(median,inplace=True)
sample_incomplete_rows

#now doing with the actual df
housing["total_bedrooms"].fillna(median,inplace=True)
#recombining
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

#playing with categorical data
housing_cat = housing["ocean_proximity"]
housing_cat.head(10)
#to turn the categorical data into integers
housing_cat_encoded, housing_categories = housing_cat.factorize()
housing_categories

"""the above transformation will create the confusion that nearby numbers
are more similar that those far by. This need not be right. So we use
one hot encoding. One attribute will be equal to 1 others 0. """
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
#reshape to 2d array
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot
#Sparse matrix to dense array
hcea = housing_cat_1hot.toarray()

#separating the categorical and numerical labels
housing_num = housing.drop('ocean_proximity', axis=1)
cat_attribs = ['ocean_proximity']
num_attribs = list(housing_num)
#performing scaling using StandardScaler
from sklearn import preprocessing
std_scaler = preprocessing.StandardScaler()
housing_num[num_attribs] = std_scaler.fit_transform(housing_num[num_attribs])
#adding numerical categorical data
enc_data = pd.DataFrame(housing_cat_1hot.toarray())
enc_data.columns = housing_categories
enc_data.index = housing.index 
housing_prepared = housing_num.join(enc_data)

#new data 
housing_prepared.head(10)
housing_prepared.describe()
housing_prepared.shape

#linear regression model 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing_prepared.iloc[:5]
some_labels = housing_labels.iloc[:5]

print("Predictions:", lin_reg.predict(some_data))

print("Labels:", list(some_labels))

#calculating rmse
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

#using cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
