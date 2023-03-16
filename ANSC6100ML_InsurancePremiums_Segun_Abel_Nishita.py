# ANSC 6100 - Final Project
# Authors: Abelhard Jauwena, Olusegun Odumosu, Nishita Sharif

import pandas
import io

# GitHub URL to .csv.
dataset_url_path = 'https://raw.githubusercontent.com/nishitas1007/ANSC6100_InsurancePremiums/main/insurance_dataset.csv'

# Assigning the .csv in the URL to a dataframe.
insurance_premium_df = pandas.read_csv(dataset_url_path)

# Use the info() function to see information about our dataframe.
insurance_premium_df.info()
# There are no "null" values in any of the 7 columns of our dataframe. As such, we don't need to remove any NAs.

# print(insurance_premium_df)

# --- Abel: These are the stuff I tried at home. Feel free to change or add anything! ---
# --- You have to download the .csv file to run this script. ---

# Import the appropriate modules.
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

# Create an argument parser.
parser = argparse.ArgumentParser(description='data quality control')

# Add arguments.
parser.add_argument('-in', '--in_file', action='store', dest='in_file', required=False, default='insurance_premium.csv', help='the name of the .csv file containing the data set of interest, where the last column is the desired output')
parser.add_argument('-k', '--num_neighbors', action='store', dest='kk', default=5, required=False, help='the number of neighbors') # Check what the default value for k is.
parser.add_argument('-mk', '--max_num_neighrbors', action='store', dest='max_kk', default=50, required=False, help='the maximum number of neighbors') # Check what the default maximum value for k is.

# Handle user errors.
try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

# Save the arguments in separate variables.
data_set = args.in_file
kk = int(args.kk)
max_kk = int(args.max_kk)

# Read the insurance premium data set as a pandas dataframe. Treat the first column as column names.
df = pd.read_csv(data_set, header=0)

# Look at the first five rows of the data frame.
print('Original Data Frame:')
print(df.head)
# The variables that must be converted into numerical data with "int" data type are:
    # "sex," "smoker," and "region" (they are categorical data).
    # "bmi" (its data type is "str").

# Convert "bmi" from "str" to "int."
df['bmi_int'] = df['bmi'].apply(lambda x: int(x)) # (Denis, 2018).

# Prepare LabelEncoder (Ray, 2015).
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

# Convert "sex," "smoker," and "region" into numerical data.
df['sex'] = le_sex.fit_transform(df['sex']) # (Denis, 2018).
df['smoker'] = le_smoker.fit_transform(df['smoker']) # (Denis, 2018).
df['region'] = le_region.fit_transform(df['region']) # (Denis, 2018).

# Define the bins for "charges" (Jain, 2020).
bins = [1121.87, 7386.73, 13651.58, 19916.44, 26181.30, 32446.15, 38711.01, 44975.86, 51240.72, 57505.574, 63770.428]

# Define the categories for "charges" corresponding to the bins above for classification.
categories = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # (Wisdom, 2019).

# Append the categories for "charges" to the dataframe.
df['charge categories'] = pd.cut(df.charges, bins=bins, labels=categories) # (Wisdom, 2019).

# Drop all NAs from the data frame.
df = df.dropna()

# Define the names of all the input variables.
inputs = ['age', 'sex', 'bmi_int', 'children', 'smoker', 'region'] # (Denis, 2018).

# Define and scale the input variables.
X = df[inputs]
sc = StandardScaler()
X = sc.fit_transform(X)

# Define the output variable.
y = df['charge categories']

# Split the data set into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)
# Try different (e.g., 80:20) train:test splits.

# Create and fit a KNN classifier.
knn_classifier = KNeighborsClassifier(n_neighbors=kk)
knn_classifier.fit(X_train, y_train)

# Test how accurately the KNN classifier predicts on the testing set.
y_pred = knn_classifier.predict(X_test)

# Print the results.
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
