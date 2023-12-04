import numpy as np
import pandas as pd


def read():
    df_er = pd.read_csv("data/yen_dollar_exchange_rate.csv", sep=",", encoding='unicode_escape')
    df_dint = pd.read_csv("data/dollar_interest_rate.csv", sep=",")
    df_yint = pd.read_csv("data/yen_interest_rate.csv", sep=",", encoding='unicode_escape')
    return df_er, df_dint, df_yint


def format(df_er:pd.DataFrame, df_dint:pd.DataFrame, df_yint:pd.DataFrame):
    """
    :param df_er: Pandas DataFrame of Yen/Dollar Exchange Rate
    :param df_dint: Pandas DataFrame of Dollar Interest Rate
    :param df_yint: Pandas DataFrame of Yen Interest Rate
    :return:
    """

    #format df_er
    df_er = df_er.dropna()
    df_er = df_er.drop(index=0)
    df_er.index = pd.to_datetime(df_er['Series code'])
    df_er.rename(columns={"FM08'FXERM07": "exchange rate"}, inplace=True)
    df_er.drop(["Series code"], axis=1, inplace=True)
    df_er["exchange rate"] = df_er["exchange rate"].astype("float")

    # format df_dint
    df_dint = df_dint.dropna()
    df_dint.index = pd.to_datetime(df_dint['DATE'])
    df_dint.drop(["DATE"], axis=1, inplace=True)

    # format df_dint
    df_yint = df_yint.dropna()
    df_yint = df_yint.drop(index=0)
    df_er.rename(columns={"IR01'MADR1M": "interest rate"}, inplace=True)
    df_yint.index = pd.to_datetime(df_yint['Series code'])
    df_yint.drop(["Series code"], axis=1, inplace=True)

    return df_er, df_dint, df_yint


def format_prophet(df:pd.DataFrame, date_start='1993-10-1', date_end='2023-09-01'):
    """
    :param df: Pandas DataFrame whose index's data type is DateTime and only column's data type is float.
    :param date_start: starting DateTime to slice the DataFrame
    :param date_end:
    :return: DataFrame
    """
    df_p = df[date_start:date_end].copy()
    df_p["ds"] = df_p.index
    df_p.reset_index(inplace=True, drop=True)
    column_name = df.columns
    df_p.rename(columns={
        column_name[0]: "y"
    }, inplace=True)
    return df_p


def format_er(df_er_prophet, df_dint_prophet, df_yint_prophet):
    """
    :param df_er_prophet: DataFrame for prophet that has ds and y as columns
    :param df_dint_prophet: DataFrame for prophet that has ds and y as columns
    :param df_yint_prophet: DataFrame for prophet that has ds and y as columns
    :return: concatinated DataFrame with ds, y (exchange rate), ir_dollar, and ir_yen of existing information
    """
    df = pd.concat([df_er_prophet, df_dint_prophet["y"].rename("ir_dollar"), df_yint_prophet["y"].rename("ir_yen")],
                   axis=1, join='inner')
    return df


def format_er_future(df_dint_prophet, forecast_dollar_ir, df_yint_prophet, forecast_yen_ir, cap=158, future_start_date='2023-09-30'):
    future = pd.DataFrame(forecast_dollar_ir['ds']).reset_index(drop=True)
    # set the upper bound to 158 yen per dollar
    future['cap'] = cap
    future['ir_dollar'] = pd.concat([df_dint_prophet['y'],
                                    forecast_dollar_ir.loc[(forecast_dollar_ir["ds"] >= future_start_date),'yhat']], axis=0)
    future['ir_yen'] = pd.concat([df_yint_prophet['y'],
                                  forecast_yen_ir.loc[(forecast_yen_ir["ds"] >= future_start_date), 'yhat']], axis=0)
    return future




