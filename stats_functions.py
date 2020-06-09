import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn import model_selection

def avg(b_list):
    n, mean = len(b_list), 0.0

    # If n is less than or equal to 1, then the mean is the first element
    if n <= 1:
        return b_list[0]

    # Calculating average
    for i in b_list:
        mean = mean + float(i)
    mean = mean / float(n)

    return mean

def sd(b_list):
    n = len(b_list)

    # If n is less than or equal to 1, then the standard deviation is 0
    if n <= 1:
        return 0.0

    mean, sd = avg(b_list), 0.0

    # Calculate standard deviation using the formula
    for i in b_list:
        sd += (float(i) - mean)**2
    sd = ((sd / float(n-1)))**(1/2)

    return sd



# bivariate regression function with optional plotting
def bivariate_model(X_input,y, plots = False):
  '''
  X_input = Independent Variable (df["col"])
  y       = Dependent Variable (df["col"])
  plots   = True or False
  Returns the MSE, R^2 Values and Model
  '''

  X = X_input[:, np.newaxis]

  # Splitting the Data into Training and Test Sets
  Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=1, test_size = 0.3)

  # Fitting the Model while defaulting the Intercept as True
  model = LinearRegression(fit_intercept = True)
  # Fitting the Model with the Training Data
  model.fit(Xtrain,ytrain)
  # Gathering the Predictions using the Held Out Test Set
  y_model = model.predict(Xtest)

  # Calculating the Mean Squared Error as an Additional Assessment Metric
  prediction_difference = y_model - ytest
  mse = (np.mean(prediction_difference))**2

  print("Mean Squared Error:", mse)
  print("Training R^2:", model.score(Xtrain,ytrain))
  print("Testing R^2:", model.score(Xtest,ytest))
  print("y = ", model.coef_[0], "*x + ", model.intercept_)
  print()

  print_on_plot_train = f'Training R^2 = {model.score(Xtrain,ytrain):0.3f} \nModel: y = {model.coef_[0]:0.3f} *X + {model.intercept_: 0.3f}'
  print_on_plot_test = f'MSE = {mse:0.3f} , \nTesting R^2 = {model.score(Xtest,ytest):0.3f} , \nModel: y = {model.coef_[0]:0.3f} *X + {model.intercept_: 0.3f}'

  if plots == True:
    
    # Two Plots Side by Side
    fig,axes = plt.subplots(nrows = 1, ncols = 2,
                            sharex = False, sharey = False,
                            figsize = (10,5))
    
    axes[0].plot(Xtrain, ytrain,
               linestyle = "",
               marker = ".",
               markerfacecolor = "Green",
               markeredgecolor = "Green")
    axes[0].plot(Xtest, y_model,
               color = "Blue")
    axes[0].set_title('Training Data')
    axes[0].text(0.8, 0.1, print_on_plot_train, horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes, fontsize=12)

    axes[1].plot(Xtest, ytest,
                linestyle = "",
                marker = ".",
                markerfacecolor = "Red",
                markeredgecolor = "Red")
    axes[1].plot(Xtest, y_model,
                color = "Blue")
    axes[1].set_title('Testing Data')
    axes[1].text(0.8, 0.1, print_on_plot_test, horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes, fontsize=12)

    plt.tight_layout()

# General Regression with Optional Summary and Cross Validation Function
def general_regression(X, y, summary = False, kfoldcv = False, splits = 10):
  '''
  X = x inputs
  y = y inputs
  summary = True or False
  kfoldcv = True or False, cv with 10 folds
  loocv = True or False
  returns MSE, R^2, and Summary
  '''
  Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,random_state=1)

  model = LinearRegression(fit_intercept=True)
  model.fit(Xtrain,ytrain)
  y_model = model.predict(Xtest)
  prediction_difference = y_model - ytest
  mse = (np.mean(prediction_difference))**2

  print("-------------Model Assessment-------------")
  print("Mean Squared Error:", mse)
  print('Training R^2',model.score(Xtrain,ytrain))
  print('Testing R^2',model.score(Xtest,ytest))
  print()

  if summary == True:
    sm_model = sm.OLS(ytrain, Xtrain)
    sm_results = sm_model.fit()
    print("------------------------------------Model Summary------------------------------------")
    print(sm_results.summary())
    print()

  if kfoldcv == True:
    kfold = model_selection.KFold(n_splits = splits)
    cv_model = LinearRegression()
    cv_results = model_selection.cross_val_score(cv_model, Xtrain, ytrain, cv = kfold, scoring='r2')
    print("-------------kfold CV Results-------------")
    print()
    print("Cross Validation results:", cv_results)
    print()

    print("Cross Validation mean:", cv_results.mean())
    print("Cross Validation std:",  cv_results.std())



