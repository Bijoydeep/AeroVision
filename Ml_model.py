from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
def predict_pollution(df):
    df = df.dropna()
    df = df.tail(30)  # last 30 days
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["value"].values
    model = LinearRegression().fit(X, y)
    future_X = np.arange(len(df), len(df)+7).reshape(-1, 1)
    future_y = model.predict(future_X)
    forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=7)
    return pd.Series(future_y, index=forecast_dates)
