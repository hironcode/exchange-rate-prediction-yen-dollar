import numpy as np
import pandas as pd
import optuna
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, f1_score
import matplotlib.pyplot as plt
import datetime as dt


"""
References:

Prophetモデルの概要とOptunaで自動でチューニングする方法
https://book.st-hakky.com/data-science/prophet-model-predicts/

"""


def visualize(df_train, df_test):
    fig, ax = plt.subplots()
    ax.plot(df_train.ds, df_train.y, label="actual(train dataset)")
    ax.plot(df_test.ds, df_test.y, label="actual(test dataset)")
    plt.legend()


def assess(model, test_length, df, df_train, df_test):
    # Prediction
    df_future = model.make_future_dataframe(periods=test_length, freq='M')
    df_pred = model.predict(df_future)

    df['Predict'] = df_pred['yhat']

    train_pred = df.iloc[:-test_length].loc[:, 'Predict']
    test_pred = df.iloc[-test_length:].loc[:, 'Predict']

    y_train = df.iloc[:-test_length].loc[:, 'y']
    y_test = df.iloc[-test_length:].loc[:, 'y']

    print('RMSE:')
    print(np.sqrt(mean_squared_error(y_test, test_pred)))
    print('MAE:')
    print(mean_absolute_error(y_test, test_pred))
    print('MAPE:')
    print(mean_absolute_percentage_error(y_test, test_pred))

    fig, ax = plt.subplots()
    ax.plot(df_train.ds, df_train.y, label="actual(train dataset)")
    ax.plot(df_test.ds, df_test.y, label="actual(test dataset)")
    ax.plot(df_train.ds, train_pred, linestyle="dotted", lw=2,color="m")
    ax.plot(df_test.ds, test_pred, label="Prophet", linestyle="dotted", lw=2, color="m")
    plt.legend()