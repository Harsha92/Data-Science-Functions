import os
import time
import collections
import copy
import requests
import sys
# -------------------------------------------------------------------------------------------------------------------------------
import matplotlib
import numpy as np
import pandas as pd  # Importing package pandas (For Panel Data Analysis)
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport  # Import Pandas Profiling (To generate Univariate Analysis)
# -------------------------------------------------------------------------------------------------------------------------------
pd.set_option('mode.chained_assignment', None)  # To suppress pandas warnings.
pd.set_option('display.max_colwidth', None)  # To display all the data in each column
pd.set_option('display.max_columns', None)  # To display every column of the dataset in head()
# -------------------------------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, \
    recall_score, fbeta_score, mean_absolute_error, mean_squared_error, r2_score

#####################################################################################
def function_list():
    '''
    The main aim of this module is to Modularize the most commonly performed activities in EDA
     and Machine Learning using the concept of functions in Python.

     ML_functions module consists of below functions

    Feature Engineering
        1. missing_values_analysis
            - Input :
                Dataframe
                missing_pcnt : missing data percentage
            - Output :
                Generates a dataframe with missing values and it's percentages in each columns.
                Also displays the columns with missing values more than missing_pcnt%

        2. corr_heatmap
            - Input :
                df : Input the dataframe on which you want to find the correlation
                corr_pcnt : Finds the columns with correlation more than corr_pcnt
            - Output :
                Heatmap and list of columns with high correlation

        3. one_hot_encoding_top_x
            function to create the dummy variables for the most frequent labels
            - Input : Dataframe, column name and top x labels
            - Output : Performs OHE on top x labels

    Machine Learning
        1. performance_metrics_classification
            - Input :   X_test, y_test,y_pred_test, model  and
                algorithm used to build the model
            - Output : Returns Confusion Matrix, Accuracy, Precision, Recall, Fbeta Score and Classification Report

        2. performance_metrics_regression
            - Input : X_test, y_test, model, y_test, predictions on y_test (y_test_pred) and
                algorithm used to build the model
            - Output : Returns Mean Absolute Error, Mean Squared Error, Root Mean Squared Error, R2 Score
                and Adjusted R2 Score

        3. hyperparameter_tuning_logistic_regression
            - Inputs:
                X_train, y_train, X_test,
                model: Basic Logistic Regression model variable name,
                n_iter: no.of iterations (default of 60),
                cv: K fold Cross Validation (default of 10),
                verbose: verbose (default of 10)
            - Output:
                y_pred_test_lr_grid : Predictions returned from Logistic Regression model using Grid Search CV technique
                lr_grid : Logistic Regression model built using Grid Search CV technique

        4. hyperparameter_tuning_decision_tree
            - Inputs:
                X_train, y_train, X_test,
                model: Basic Descison Tree model variable name,
                n_iter: no.of iterations (default of 60),
                cv: K fold Cross Validation (default of 10),
                verbose: verbose (default of 10)
            - Output:
                y_pred_test_dt_grid : Predictions returned from Decision Tree model using Grid Search CV technique
                dt_grid : Decision Tree model built using Grid Search CV technique

        5. hyperparameter_tuning_random_forest
            - Inputs:
                X_train, y_train, X_test,
                model: Basic Random Forest model variable name,
                n_iter: no.of iterations (default of 20),
                cv: K fold Cross Validation (default of 5),
                verbose: verbose (default of 10)
            - Output:
                y_pred_test_rf_grid : Predictions returned from Random Forest model using Grid Search CV technique
                rf_grid : Random Forest built using Grid Search CV technique

    '''
    pass


################ ######  FEATURE ENGINEERING #############################################

def corr_heatmap(df, corr_pcnt=0.8):
    '''
    Input :
                df : Input the dataframe on which you want to find the correlation
                corr_pcnt : Finds the columns with correlation more than corr_pcnt
    Output : Heatmap and list of columns with high correlation
    '''
    # Calculate pairwise-correlation
    matrix = df.corr()

    # Create a mask
    mask = np.triu(np.ones_like(matrix, dtype=bool))

    # Create a custom diverging palette
    cmap = sns.diverging_palette(250, 15, s=75, l=40,
                                 n=9, center="light", as_cmap=True)

    plt.figure(figsize=(20, 12))

    print(sns.heatmap(matrix, mask=mask, center=0, annot=True,
                      fmt='.2f', square=True, cmap=cmap))

    global highcorrel
    highcorrel = set()
    correlmatrix = df.corr()

    for x in range(len(correlmatrix.columns)):
        for y in range(x):
            if abs(correlmatrix.iloc[x, y]) > corr_pcnt:
                colname = correlmatrix.columns[y]
                highcorrel.add(colname)

    print('The features with correlation more than {0}% are as below {1}\n'.format(corr_pcnt, highcorrel))


#####################################################################################

def missing_values_analysis(data, missing_pcnt=70):
    '''
    Input :
                Dataframe
                missing_pcnt : missing data percentage
    Output :
                Generates a dataframe with missing values and it's percentages in each columns.
                Also displays the columns with missing values more than missing_pcnt%
    '''
    global missing_data_columns, top_missing_data_columns
    missing_data = pd.DataFrame(data.isnull().sum(), columns=['Total Missing Values'])
    missing_data['% of Missing Values'] = data.isnull().sum() / len(data) * 100
    missing_data = missing_data.sort_values(by='% of Missing Values', ascending=False)
    print(missing_data)
    print('-----------------------------------------------------------------------------------------------------------------')

    print('Below are the columns with more than {}% of missing data'.format(missing_pcnt))
    top_missing_data_columns = missing_data[missing_data['% of Missing Values'] > missing_pcnt]
    print(top_missing_data_columns)
    print('We can access the missing value columns more than {}% from top_missing_data_columns variable'.format(missing_pcnt))

    print('-----------------------------------------------------------------------------------------------------------------')
    # Finding the missing values in remaining features
    print('Below are the rest of the columns with missing data less than {}%'.format(missing_pcnt))
    missing_data_columns = data.columns[(data.isnull().sum() > 0) & (data.isnull().sum() < missing_pcnt)]
    print(missing_data_columns)
    print('We can access the remaining missing data columns from  missing_data_columns variable')

####################################################################################

# get whole set of dummy variables, for all the categorical variables

def one_hot_encoding_top_x(df, variable, top_x_labels):
    '''
     function to create the dummy variables for the most frequent labels
     Input : Dataframe, column name and top x labels
     output : Performs OHE on top x labels

    Calculation of top_x_labels
    top_x_labels = [y for y in df.variable.value_counts().sort_values(ascending=False).head(x).index]

    '''

    for label in top_x_labels:
        df[variable + '_' + label] = np.where(df[variable] == label, 1, 0)


##########################  MACHINE LEARNING ##########################################

def performance_metrics_classification(X_test, y_test, y_pred_test, *, model, algorithm, beta=1):
    """
    Input :   X_test, y_test,y_pred_test, model  and
                algorithm used to build the model

    Output : Returns Confusion Matrix, Accuracy, Precision, Recall, Fbeta Score and Classification Report
    """
    global Results_dict
    print('Performance metrics for {} for test data are as below '.format(algorithm))
    sns.set(context="paper", font_scale=1.5)
    cm = confusion_matrix(y_test, y_pred_test)
    ax = heatmap = sns.heatmap(cm, cmap="Blues", annot=True, fmt=".0f")
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Outcome')
    plt.xlabel('Predicted Outcome')
    plt.yticks(rotation=0)
    plt.show()

    print('Accuracy score for test data is:', accuracy_score(y_test, y_pred_test))
    print('Precession score for test data is:', precision_score(y_test, y_pred_test))
    print('Recall score for test data is:', recall_score(y_test, y_pred_test))
    print('F{} score for test data is: {}'.format(beta, fbeta_score(y_test, y_pred_test, beta=beta)))
    print()
    print('Classification Report for test data is: \n', classification_report(y_test, y_pred_test))

    Results_dict = {
        'Algorithm': algorithm,
        'Accuracy': accuracy_score(y_test, y_pred_test),
        'Precision': precision_score(y_test, y_pred_test),
        'Recall': recall_score(y_test, y_pred_test),
        'Fbeta Score': fbeta_score(y_test, y_pred_test, beta=beta)
    }


#####################################################################################

def performance_metrics_regression(X_test, y_test, y_pred_test, *, model, algorithm):
    """
   Input : X_test, y_test, model, y_test, predictions on y_test (y_test_pred) and
                algorithm used to build the model
    Output : Returns Mean Absolute Error, Mean Squared Error, Root Mean Squared Error, R2 Score
                and Adjusted R2 Score
    """
    global Results_dict
    print('Performance metrics for {} are as below '.format(algorithm))

    print('Mean Absolute Error for test data is :', mean_absolute_error(y_test, y_pred_test))
    print('Mean Squared Error for test data is :', mean_squared_error(y_test, y_pred_test))
    print('Root Mean Square Error for test data is :', np.sqrt(mean_squared_error(y_test, y_pred_test)))
    print('R2 Score for test data is :', r2_score(y_test, y_pred_test))
    adjusted_r_sqaure = 1 - (1 - r2_score(y_test, y_pred_test)) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
    print('Adjusted R2 Score of test data is :', adjusted_r_sqaure)

    Results_dict = {
        'Algorithm': algorithm,
        'MAE': mean_absolute_error(y_test, y_pred_test),
        'MSE': mean_squared_error(y_test, y_pred_test),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'R2 Score': r2_score(y_test, y_pred_test),
        'Adj R2 Score' : adjusted_r_sqaure
    }

#####################################################################################

def hyperparameter_tuning_logistic_regression(X_train, y_train, X_test, penalty=None, solver=None, C=None, dual=None,
                                              tol=None, max_iter=None, *, model, n_iter=60, cv=10, verbose=10):
    '''
    Inputs:
    X_train, y_train, X_test,
    model: Basic Logistic Regression model variable name,
    n_iter: no.of iterations (default of 60),
    cv: K fold Cross Validation (default of 10),
    verbose: verbose (default of 10)

    Hyperparameters with default values as follows
    penalty = ['l1', 'l2', 'elasticnet']
    solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    C = [100, 10, 1.0, 0.1, 0.01]
    dual = [True, False]
    tol = [1e-4, 1e-3, 1e-2, 1e-5, 1e-6]
    max_iter = [25, 50, 75, 100, 125, 150, 175, 200]

    Processing:
    Builds a model with above Hyperparameters usig Randomized Search CV technique. Then it computes the best
    parameters based on Randomized Search CV model. After obtaining the best parameters we will pass these
    parameters to Grid Search CV by including some varriance around these parameters so that Grid Search CV can
    search the entire grid and returns the best parameters.
    We will use these parameters to build a model and perform predictions on test data

    Output:
     y_pred_test_lr_grid : Predictions returned from Logistic Regression model using Grid Search CV technique
     lr_grid : Logistic Regression model built using Grid Search CV technique

    '''
    global random_grid_lr, lr_random, grid_search_lr, lr_grid, y_pred_test_lr_grid
    if penalty is None:
        penalty = ['l1', 'l2', 'elasticnet']

    if solver is None:
        solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

    if C is None:
        C = [100, 10, 1.0, 0.1, 0.01]

    if dual is None:
        dual = [True, False]

    if tol is None:
        tol = [1e-4, 1e-3, 1e-2, 1e-5, 1e-6]

    if max_iter is None:
        max_iter = [25, 50, 75, 100, 125, 150, 175, 200]

    random_grid_lr = {'penalty': penalty,
                      'solver': solver,
                      'C': C,
                      'dual': dual,
                      'tol': tol,
                      'max_iter': max_iter
                      }

    # Random search of parameters, using k fold cross validation,

    lr_random = RandomizedSearchCV(estimator=model,
                                   param_distributions=random_grid_lr,
                                   n_iter=n_iter, cv=cv, verbose=verbose, random_state=0, n_jobs=-1)
    lr_random.fit(X_train, y_train)
    print('\nThe best parameters from Randomized Search CV are as below \n', lr_random.best_params_)
    print('\nThe best estimator from Randomized Search CV are as below \n', lr_random.best_estimator_)
    print('\nThe best Score from Randomized Search CV is :', lr_random.best_score_)
    print()

    grid_search_lr = {'tol': [lr_random.best_params_['tol']],
                      'solver': [lr_random.best_params_['solver']],
                      'penalty': [lr_random.best_params_['penalty']],
                      'max_iter': [lr_random.best_params_['max_iter'],
                                   lr_random.best_params_['max_iter'] + 20,
                                   lr_random.best_params_['max_iter'] - 20,
                                   lr_random.best_params_['max_iter'] + 40,
                                   lr_random.best_params_['max_iter'] - 40],
                      'dual': [lr_random.best_params_['dual']],
                      'C': [lr_random.best_params_['C'],
                            lr_random.best_params_['C'] + 1,
                            lr_random.best_params_['C'] + 2,
                            lr_random.best_params_['C'] + 3,
                            lr_random.best_params_['C'] + 4]
                      }

    lr_grid = GridSearchCV(estimator=model, param_grid=grid_search_lr,
                           cv=cv, verbose=verbose, n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    print('\nThe best parameters from Grid Search CV are as below \n', lr_grid.best_params_)
    y_pred_test_lr_grid = lr_grid.predict(X_test)
    print('\nWe can access predictions on test data from y_pred_test_lr_grid variable and the Logistic Regression model'
          'form lr_grid variable')

#####################################################################################

def hyperparameter_tuning_decision_tree(X_train, y_train, X_test, criterion=None, splitter=None, max_depth=None,
                                        min_samples_split=None, min_samples_leaf=None, max_features=None, *,
                                        model, n_iter=60, cv=10, verbose=10):
    '''
    Inputs:
    X_train, y_train, X_test,
    model: Basic Descison Tree model variable name,
    n_iter: no.of iterations (default of 60),
    cv: K fold Cross Validation (default of 10),
    verbose: verbose (default of 10)

    Hyperparameters with default values as follows
    criterion = ['gini', 'entropy']
    splitter = ['best', 'random']
    max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
    min_samples_split = [2, 5, 10, 15, 100]
    min_samples_leaf = [1, 2, 5, 10]
    max_features = ['auto', 'sqrt', 'log2', 'None']

    Processing:
    Builds a model with above Hyperparameters usig Randomized Search CV technique. Then it computes the best
    parameters based on Randomized Search CV model. After obtaining the best parameters we will pass these
    parameters to Grid Search CV by including some varriance around these parameters so that Grid Search CV can
    search the entire grid and returns the best parameters.
    We will use these parameters to build a model and perform predictions on test data

    Output:
     y_pred_test_dt_grid : Predictions returned from Decision Tree model using Grid Search CV technique
     dt_grid : Decision Tree model built using Grid Search CV technique

    '''
    global y_pred_test_dt_grid, dt_grid
    if criterion is None:
        criterion = ['gini', 'entropy']

    if splitter is None:
        splitter = ['best', 'random']

    if max_depth is None:
        max_depth = [int(x) for x in np.linspace(5, 30, num=6)]

    if min_samples_split is None:
        min_samples_split = [2, 5, 10, 15, 100]

    if min_samples_leaf is None:
        min_samples_leaf = [1, 2, 5, 10]

    if max_features is None:
        max_features = ['auto', 'sqrt', 'log2', 'None']

    random_grid_dt = {'criterion': criterion,
                      'splitter': splitter,
                      'max_depth': max_depth,
                      'min_samples_split': min_samples_split,
                      'min_samples_leaf': min_samples_leaf,
                      'max_features': max_features
                      }

    # Random search of parameters, using k fold cross validation,

    dt_random = RandomizedSearchCV(estimator=model,
                                   param_distributions=random_grid_dt,
                                   n_iter=n_iter, cv=cv, verbose=verbose, random_state=0, n_jobs=-1)
    dt_random.fit(X_train, y_train)
    print('\nThe best parameters from Randomized Search CV are as below \n', dt_random.best_params_)
    print('\nThe best estimator from Randomized Search CV are as below \n', dt_random.best_estimator_)
    print('\nThe best Score from Randomized Search CV is :', dt_random.best_score_)
    print()

    grid_search_dt = {'splitter': [dt_random.best_params_['splitter']],
                      'min_samples_split': [dt_random.best_params_['min_samples_split'],
                                            dt_random.best_params_['min_samples_split'] + 25,
                                            dt_random.best_params_['min_samples_split'] - 25,
                                            dt_random.best_params_['min_samples_split'] + 50,
                                            dt_random.best_params_['min_samples_split'] - 50],
                      'min_samples_leaf': [dt_random.best_params_['min_samples_leaf'],
                                           dt_random.best_params_['min_samples_leaf'] + 1,
                                           dt_random.best_params_['min_samples_leaf'] + 2,
                                           dt_random.best_params_['min_samples_leaf'] + 4,
                                           dt_random.best_params_['min_samples_leaf'] + 5],
                      'max_features': [dt_random.best_params_['max_features']],
                      'max_depth': [dt_random.best_params_['max_depth'],
                                    dt_random.best_params_['max_depth'] + 25,
                                    dt_random.best_params_['max_depth'] + 50,
                                    dt_random.best_params_['max_depth'] + 75,
                                    dt_random.best_params_['max_depth'] + 100],
                      'criterion': [dt_random.best_params_['criterion']]

                      }

    dt_grid = GridSearchCV(estimator=model, param_grid=grid_search_dt, cv=cv,
                           verbose=verbose, n_jobs=-1)
    dt_grid.fit(X_train, y_train)
    print('\nThe best parameters from Grid Search CV are as below \n', dt_grid.best_params_)
    y_pred_test_dt_grid = dt_grid.predict(X_test)
    print('\nWe can access predictions on test data from y_pred_test_dt_grid variable and the Decision tree model'
          'form dt_grid variable')


def hyperparameter_tuning_random_forest(X_train, y_train, X_test, n_estimators=None, max_features=None,
                                        max_depth=None, min_samples_split=None, min_samples_leaf=None, *,
                                        model, n_iter=20, cv=5, verbose=10):

    '''
        Inputs:
        X_train, y_train, X_test,
        model: Basic Random Forest model variable name,
        n_iter: no.of iterations (default of 20),
        cv: K fold Cross Validation (default of 5),
        verbose: verbose (default of 10)

        Hyperparameters with default values as follows
        n_estimators = [int(x) for x in np.linspace(start = 50, stop = 300, num = 6)]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]

        Processing:
        Builds a model with above Hyperparameters usig Randomized Search CV technique. Then it computes the best
        parameters based on Randomized Search CV model. After obtaining the best parameters we will pass these
        parameters to Grid Search CV by including some varriance around these parameters so that Grid Search CV can
        search the entire grid and returns the best parameters.
        We will use these parameters to build a model and perform predictions on test data

        Output:
         y_pred_test_rf_grid : Predictions returned from Random Forest model using Grid Search CV technique
         rf_grid : Random Forest built using Grid Search CV technique

    '''
    global y_pred_test_rf_grid, rf_grid

    if n_estimators is None:
        n_estimators = [int(x) for x in np.linspace(start = 50, stop = 300, num = 6)]

    if max_features is None:
        max_features = ['auto', 'sqrt']

    if max_depth is None:
        max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]

    if min_samples_split is None:
        min_samples_split = [2, 5, 10]

    if min_samples_leaf is None:
        min_samples_leaf = [1, 2, 4]

    random_grid_rf = {'n_estimators': n_estimators,
                      'max_features': max_features,
                      'max_depth': max_depth,
                      'min_samples_split': min_samples_split,
                      'min_samples_leaf': min_samples_leaf
                      }

    # Random search of parameters, using 5 fold cross validation

    rf_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid_rf,
                                   n_iter=n_iter, cv=cv, verbose=verbose, n_jobs=-1)
    rf_random.fit(X_train, y_train)
    print('\nThe best parameters from Randomized Search CV are as below \n', rf_random.best_params_)
    print('\nThe best estimator from Randomized Search CV are as below \n', rf_random.best_estimator_)
    print('\nThe best Score from Randomized Search CV is :', rf_random.best_score_)
    print()

    grid_search_rf = {'n_estimators': [rf_random.best_params_['n_estimators'],
                                       rf_random.best_params_['n_estimators'] + 50,
                                       rf_random.best_params_['n_estimators'] - 50],
                      'min_samples_split': [rf_random.best_params_['min_samples_split'],
                                            rf_random.best_params_['min_samples_split'] - 2,
                                            rf_random.best_params_['min_samples_split'] + 2],
                      'min_samples_leaf': [rf_random.best_params_['min_samples_leaf'],
                                           rf_random.best_params_['min_samples_leaf'] + 2,
                                           rf_random.best_params_['min_samples_leaf'] + 4],
                      'max_features': [rf_random.best_params_['max_features']],
                      'max_depth': [rf_random.best_params_['max_depth'],
                                    rf_random.best_params_['max_depth'] + 15,
                                    rf_random.best_params_['max_depth'] + 20]

                      }

    rf_grid = GridSearchCV(estimator=model, param_grid=grid_search_rf, cv=cv,
                           verbose=verbose, n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    print('\nThe best parameters from Grid Search CV are as below \n', dt_grid.best_params_)
    y_pred_test_rf_grid = rf_grid.predict(X_test)
    print('\nWe can access predictions on test data from y_pred_test_rf_grid variable and the Decision tree model'
          'form rf_grid variable')


