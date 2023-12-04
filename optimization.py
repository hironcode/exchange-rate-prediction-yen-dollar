import numpy as np
import pandas as pd
import optuna
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


"""
References:
https://book.st-hakky.com/data-science/prophet-model-predicts/
https://tech.515hikaru.net/post/2019-06-26-optuna-have-arg/
"""


# setting an objective function inside a customizable higher-order function for interest rate prediction models
def _customize_objective(df_train, df_test, test_length):
    def _objective(trial):
        params = {
            'changepoint_prior_scale': trial.suggest_uniform('changepoint_prior_scale', 0.001, 0.8),
            'seasonality_prior_scale': trial.suggest_uniform('seasonality_prior_scale', 0.01, 10),
            'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
            'changepoint_range': trial.suggest_discrete_uniform('changepoint_range', 0.8, 0.95, 0.001),
            'n_changepoints': trial.suggest_int('n_changepoints', 20, 35),
        }
        m = Prophet(**params)
        m.fit(df_train)
        df_future = m.make_future_dataframe(periods=test_length,freq='M')
        df_pred = m.predict(df_future)
        preds = df_pred.tail(len(df_test))

        val_rmse = mean_squared_error(df_test.y, preds.yhat)
        return val_rmse
    return _objective


# returns a study object of Optuna
def get_best_params(df_train, df_test, test_length):
    study = optuna.create_study(direction="minimize")
    study.optimize(_customize_objective(df_train, df_test), n_trials=200)
    return study


# objective function and higher-order function for exchange rate prediction model
def _customize_objective_ex(df_train, df_test, test_length):
    def _objective_ex(trial):
        params = {
            'changepoint_prior_scale': trial.suggest_uniform('changepoint_prior_scale', 0.001, 0.8),
            'seasonality_prior_scale': trial.suggest_uniform('seasonality_prior_scale', 0.01, 10),
            'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
            'changepoint_range': trial.suggest_discrete_uniform('changepoint_range', 0.8, 0.95, 0.001),
            'n_changepoints': trial.suggest_int('n_changepoints', 20, 35),
        }
        m = Prophet(**params)
        m.add_regressor('ir_dollar')
        m.add_regressor('ir_yen')
        m.fit(df_train)
        df_future = m.make_future_dataframe(periods=test_length,freq='M')
        df_future['ir_dollar'] = pd.concat([df_train['ir_dollar'], df_test['ir_dollar']], axis=0)
        df_future['ir_yen'] = pd.concat([df_train['ir_yen'], df_test['ir_yen']], axis=0)
        df_future["cap"] = 158
        df_pred = m.predict(df_future)
        preds = df_pred.tail(len(df_test))

        val_rmse = np.sqrt(mean_squared_error(df_test.y, preds.yhat))
        return val_rmse
    return _objective_ex


# returns a study object
def get_best_params_ex(df_train, df_test):
    study = optuna.create_study(direction="minimize")
    study.optimize(_customize_objective_ex(df_train, df_test), n_trials=200)
    return study


def split_train_test(df, test_size:int):
    """
    :param df: DataFrame
    :param test_size: the number of the latest indexes to sample test data
    :return: train df and test df
    """
    df_train = df.iloc[:-test_size]
    df_test = df.iloc[-test_size:]
    return df_train, df_test


