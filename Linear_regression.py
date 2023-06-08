# -*- coding: utf-8 -*-
"""
Created on Sun May 28 14:34:42 2023

@author: XuebinLi
"""
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

#read csv
df = pd.read_csv('cleaned_final.csv',index_col=None)


def linear_regression_importance(X_train,y_train,X_test,y_test):
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    import numpy as np
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print(f'model score on training data: {model.score(X_train, y_train)}')
    print(f'model score on testing data: {model.score(X_test, y_test)}')
    
    importances = model.coef_
    indices = np.argsort(importances)
    
    fig, ax = plt.subplots()
    ax.barh(range(len(importances)), importances[indices])
    ax.set_yticks(range(len(importances)))
    _ = ax.set_yticklabels(np.array(X_train.columns)[indices])
    ax.set_title("Macro Economic Feature Importances") 


def random_forest_features_importance(X_train,y_train,X_test,y_test):
    from sklearn.ensemble import RandomForestRegressor
    
    model = RandomForestRegressor()
    
    model.fit(X_train, y_train)    
    print(f'model score on training data: {model.score(X_train, y_train)}')
    print(f'model score on testing data: {model.score(X_test, y_test)}')
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    fig, ax = plt.subplots()
    ax.barh(range(len(importances)), importances[indices])
    ax.set_yticks(range(len(importances)))
    _ = ax.set_yticklabels(np.array(X_train.columns)[indices])
    ax.set_title("Macro Economic Feature Importances")



def features_importance(X_train,y_train):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    X_train_scaled = ss.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    importances = pd.DataFrame(data={
        'Attribute': X_train.columns,
        'Importance': model.coef_[0]
    })
    importances = importances.sort_values(by='Importance', ascending=False)    
    
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
    plt.title('Feature Importances Obtained from Coefficients', size=20)
    plt.xticks(rotation=45, ha='right')  # Rotate and align x-axis labels
    plt.xlabel('Attributes')  # Add x-axis label
    plt.ylabel('Importance')  # Add y-axis label
    plt.tight_layout()  # Ensure labels do not overlap
    plt.show()


def ordinal_encoding(data):
    # Assuming 'df' is your dataframe with a date column
    data = data.reset_index() 
    
    date_column = df['Unnamed: 0']
    
    # Convert the date column to datetime type
    date_column = pd.to_datetime(date_column)
    
    # Initialize the OrdinalEncoder
    encoder = OrdinalEncoder()
    
    # Reshape the date column as a 2D array for encoding
    date_values = date_column.values.reshape(-1, 1)
    
    # Perform ordinal encoding
    encoded_dates = encoder.fit_transform(date_values)
    
    # Replace the original date column with the encoded values
    data['Date'] = encoded_dates
    data = data.drop(columns=['Unnamed: 0'])
    data = data.drop(columns=['index'])
    return data


def correlation_matrix(df):   
    plt.figure(figsize = (6, 6))
    heatmap = sns.heatmap(df.corr(), vmin = -1, vmax = 1, annot = True)
    heatmap.set_title('Macro Correlation Heatmap', fontdict = {'fontsize' : 18}, pad = 12)
    return heatmap

def clustermap(df):
    plt.figure(figsize = (4, 4))
    clustermap = sns.clustermap(df.corr(), vmin = -1, vmax = 1, annot = True)    
    return clustermap

def logistic_regression(df):
    train=df
    train['1_0'] = (train['S&P500'] - train['S&P500'].shift(1))/df['S&P500'].shift(1)
    train['1_0'] = np.where(train['1_0']>=0,1,0)
    train = train.drop('S&P500',axis=1)
    X_train, X_test, y_train, y_test = train_test_split(train.drop(['1_0','Date'],axis=1), 
                                                        train['1_0'], test_size=0.4, shuffle=True)
    logmodel = LogisticRegression()
    logmodel.fit(X_train,y_train)
    predictions = logmodel.predict(X_test)    
    from sklearn.metrics import classification_report
    print(classification_report(y_test,predictions))                               
    features_importance(X_train,y_train)

def linear_regression2(df):
    x = df.drop(['S&P500','Date'],axis=1)
    y = df['S&P500']
    # importing train_test_split from sklearn
    from sklearn.model_selection import train_test_split
    # splitting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, shuffle=True)
    # importing module
    from sklearn.linear_model import LinearRegression
    # creating an object of LinearRegression class
    LR = LinearRegression()
    # fitting the training data
    LR.fit(x_train,y_train)
    y_prediction =  LR.predict(x_test)
    # importing r2_score module

    from sklearn.metrics import r2_score
    
    from sklearn.metrics import mean_squared_error
    
    # predicting the accuracy score
    
    score=r2_score(y_test,y_prediction)
    
    print('r2 socre is ',score)
    
    print('mean_sqrd_error is==',mean_squared_error(y_test,y_prediction))
    
    print('root_mean_squared error of is==',np.sqrt(mean_squared_error(y_test,y_prediction)))
    
    #random forest features
    #random_forest_features_importance(x_train,y_train,x_test,y_test)
    linear_regression_importance(x_train,y_train,x_test,y_test)

def ridge_regression(df):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    X = df.drop(['S&P500','Date'],axis=1)
    y = df['S&P500']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
    
    # Perform feature scaling for multicollinearity
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create a Ridge regression model
    alpha = 0.5  # Regularization strength (alpha = 0 corresponds to ordinary least squares)
    ridge_model = Ridge(alpha=alpha)
    
    # Fit the model to the training data
    ridge_model.fit(X_train_scaled, y_train)
    
    # Predict on the training and testing data
    y_train_pred = ridge_model.predict(X_train_scaled)
    y_test_pred = ridge_model.predict(X_test_scaled)
    
    # Evaluate the model using mean squared error
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    
    # Calculate R-squared score
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    print(f"Ridge Regression with alpha={alpha}")
    print(f"Train MSE: {mse_train:.2f}")
    print(f"Test MSE: {mse_test:.2f}")  
    print(f"Train R-squared: {r2_train:.2f}")
    print(f"Test R-squared: {r2_test:.2f}")
    rmse = np.sqrt(mse_test)
    print(f"Root Mean Square Error for test (RMSE): {rmse:.2f}")
    
    
    # Get feature importances
    feature_importances = ridge_model.coef_
    # Sort feature importances
    indices = np.argsort(feature_importances)[::-1]
    sorted_features = [X.columns[i] for i in indices]
    sorted_importances = feature_importances[indices]
    
    # Plot feature importances
    fig, ax = plt.subplots()
    ax.barh(range(len(sorted_importances)), sorted_importances, align='center')
    ax.set_yticks(range(len(sorted_importances)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Ridge Regression - Feature Importances')
    
    plt.show()    
    
    

#main
correlation_matrix(df)
#clustermap(df)
df = ordinal_encoding(df)
#logistic_regression(df)    
#linear_regression2(df)    
ridge_regression(df)    
    
    
    