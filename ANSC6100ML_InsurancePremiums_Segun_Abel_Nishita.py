# ANSC 6100 Machine Learning Assignment
# Authors: Abelhard Jauwena, Olusegun Odumosu, Nishita Sharif

import pandas
import io

# GitHub URL to .csv
dataset_url_path = 'https://raw.githubusercontent.com/nishitas1007/ANSC6100_InsurancePremiumML/master/insurance_dataset.csv'

# Assigning the .csv in the URL to a dataframe 
insurance_premium_df = pandas.read_csv(dataset_url_path)

# use the info() function to see information about our data frame
insurance_premium_df.info()
# There are no "null" values in any of the 7 columns of our data frame – so we don't need to remove any NAs

# print(insurance_premium_df)