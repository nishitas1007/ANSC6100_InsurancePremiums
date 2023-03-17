# ANSC 6100 Machine Learning Assignment
# Authors: Abelhard Jauwena, Olusegun Odumosu, Nishita Sharif

import pandas
import csv
import io

# Using the pandas module to import the .csv file
insurance_premium_df = pandas.read_csv('/scratch/nishitas/ANSC6100_Project/insurance_dataset.csv')

# Use the info() function to see information about our data frame
insurance_premium_df.info()



