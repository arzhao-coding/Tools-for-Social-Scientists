"""Script to Run My Full Data Analysis and Project"""

# This adds the directory above to our Python path
#   This is so that we can add import our custom python module code into this script
import sys
sys.path.append('../')

# Imports
from my_module import stats_functions as stats
from my_module import test_functions as tests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn import model_selection

### I have expanded upon my work in the Computational Social Science Minor to build several functions to gather some basic statistics, conduct simple bivariate regression with optional plotting, and engage in supervised machine learning with optional summary details and/or k-fold cross validation. My interest for this project derives from my time in R where the functions for conducting these statistics are much more straightforward. Therefore, I'm hoping to replicate some of those functions in Python for future use should I find myself entering back into python. All in all, I have written four functions, two simple and two complex. 

### PYTHON SCRIPT STARTS HERE

## Using Basic Averaging Function in lieu of np.mean
# Initializing a Test List
test_list = [1, 2, 3, 4 ,5]

# The avg function computes the average of a given list or dataframe column
stats.avg(test_list)

# Testing if the results are consistent with the traditional np.mean() function
tests.test_avg(test_list)

## Using the Created Standard Deviation function in lieu of np.std() BROKEN
# Using the same test_list, using the written function
stats.sd(test_list)

# I know this function is bugged so what is the correct std according to numpy
np.std(test_list)

# Testing if the results are consistnent with the traditional np.std() function
tests.test_sd(test_list)

## Simple Data Analysis using two datasets from the seaborn module
## Bivariate Regression of Flights and Years
# Loading the Flights Dataset
flights = sns.load_dataset("flights")

# Examining the First 12 Rows
flights.head(12)

# Using the bivariate model on year vs passengers
stats.bivariate_model(flights["year"], flights["passengers"], plots = True)

# Using the general_regression function for the same variables
stats.general_regression(flights[["year"]], flights["passengers"], summary = True, kfoldcv = True, splits = 12)

## "Multivariate" Regression of Planetary Data
# Loading Data and Dropping NaNs
planets = sns.load_dataset("planets")

# Dropping missing data
planets = planets.dropna()

# Examing the first 10 Rows
planets.head(10)

# Using the bivariate model on mass vs orbital_period
stats.bivariate_model(planets["mass"], planets["orbital_period"], plots = True)

# Using the bivariate model on distance vs orbital_period
stats.bivariate_model(planets["distance"], planets["orbital_period"], plots = True)

# Using the general regression model on mass and distance to regress orbital_period
stats.general_regression(planets[["mass", "distance"]], planets["orbital_period"], summary = True, kfoldcv = True, splits = 9)