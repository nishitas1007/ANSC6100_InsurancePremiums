# to execute this program, run:
    # python final_project_draft.py

# --- IMPORTING MODULES ---
import pandas as pd
import numpy as np
import argparse
import sys
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# --- PARSING ARGUMENTS ---

# create an argument parser.
parser = argparse.ArgumentParser(description='data quality control')

# add arguments.
parser.add_argument('-in', '--in_file', action='store', dest='in_file', required=False, default='insurance_premium.csv', help='the name of the .csv file containing the data set of interest, where the last column is the desired output')
parser.add_argument('-k', '--num_neighbors', action='store', dest='kk', default=5, required=False, help='the number of neighbors') # check what the default value for k is.
parser.add_argument('-mk', '--max_num_neighrbors', action='store', dest='max_kk', default=50, required=False, help='the maximum number of neighbors') # check what the default maximum value for k is.

# handle user errors.
try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

# save the arguments in separate variables.
data_set = args.in_file
kk = int(args.kk)
max_kk = int(args.max_kk)

# --- DATA PREPROCESSING ---

# read the insurance premium data set as a pandas data frame. treat the first column as column names.
df = pd.read_csv('insurance_dataset.csv', header=0)

# look at the first five rows of the data frame.
#print('Original Data Frame:')
#print(df.info)
# variables that must be converted into numerical data with "int" data type:
    # "sex," "smoker," and "region" (categorical data).
    # "bmi" ("str" data type).

# convert "bmi" from "str" to "int."
df['bmi_int'] = df['bmi'].apply(lambda x: int(x))

# prepare LabelEncoder (Ray, 2015).
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

# convert "sex," "smoker," and "region" into numerical data.
df['sex'] = le_sex.fit_transform(df['sex']) # (Denis, 2018).
df['smoker'] = le_smoker.fit_transform(df['smoker']) # (Denis, 2018).
df['region'] = le_region.fit_transform(df['region']) # (Denis, 2018).

# define the bins for "charges" (Jain, 2020).
bins = [1121.87, 7386.73, 13651.58, 19916.44, 26181.30, 32446.15, 38711.01, 44975.86, 51240.72, 57505.574, 63770.428]

# define the categories for "charges" corresponding to the bins above for classification.
categories = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # (Wisdom, 2019).

# append the categories for "charges" to the data frame.
df['charge categories'] = pd.cut(df.charges, bins=bins, labels=categories) # (Wisdom, 2019).

# drop all NAs from the data frame.
df = df.dropna()

# define the names of all the input variables.
inputs = ['age', 'sex', 'bmi_int', 'children', 'smoker', 'region'] # (Denis, 2018).

# define the input variables.
X = df[inputs]

# scale the input variables.
sc = StandardScaler()
X = sc.fit_transform(X)

# define the output variable.
y = df['charge categories']

# split the data set into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)
# try both 80:20 and 70:30 train:test splits.

# --- BUILDING AND TESTING A KNN CLASSIFIER ---

# create a knn classifier.
knn_classifier = KNeighborsClassifier(n_neighbors=kk)

# fit the knn classifier.
knn_classifier.fit(X_train, y_train)

# test how accurately can the knn classifier predict on the testing set.
y_pred = knn_classifier.predict(X_test)

# print the results.
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))