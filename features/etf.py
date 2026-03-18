import numpy as np
import pandas as pd


def compute_etf_features(etf_day: pd.DataFrame) -> pd.DataFrame:
    """Compute ETF self features using only t and past information."""

    # Compute past-k returns from close prices.
    features = etf_day.loc[:, ["DateTime", "Close", "Vol", "Amount", "Date"]].copy()
    features["ret_1m"] = features["Close"] / features["Close"].shift(1) - 1.0
    features["ret_5m"] = features["Close"] / features["Close"].shift(5) - 1.0
    features["ret_10m"] = features["Close"] / features["Close"].shift(10) - 1.0
    features["ret_30m"] = features["Close"] / features["Close"].shift(30) - 1.0
    # Replace inf values from rare bad ticks with NaN.
    features = features.replace([np.inf, -np.inf], np.nan)

    # Compute 1m amount change as a turnover proxy.
    features["amt_chg_1m"] = features["Amount"] / features["Amount"].shift(1) - 1.0

    # Compute realized volatility over short windows from 1m returns.
    ret_sq = features["ret_1m"] * features["ret_1m"]
    features["rv_10"] = np.sqrt(ret_sq.rolling(window=10, min_periods=10).sum())
    features["rv_20"] = np.sqrt(ret_sq.rolling(window=20, min_periods=20).sum())

    # Compute rolling volume/amount stats as liquidity proxies.
    features["vol_roll_mean_20"] = features["Vol"].rolling(window=20, min_periods=20).mean()
    features["vol_roll_std_20"] = features["Vol"].rolling(window=20, min_periods=20).std()
    features["amt_roll_mean_20"] = features["Amount"].rolling(window=20, min_periods=20).mean()
    features["amt_roll_std_20"] = features["Amount"].rolling(window=20, min_periods=20).std()

    # Keep only feature columns needed downstream.
    features = features.drop(columns=["Close", "Vol", "Amount"])
    return features


def compute_label_from_close(etf_day: pd.DataFrame, horizon_minutes: int) -> pd.Series:
    """Compute forward return label from ETF close prices."""

    # Use shift(-horizon) to align Close_{t+h} with t.
    close_t = etf_day["Close"].astype(float)
    close_future = close_t.shift(-horizon_minutes)
    label = close_future / close_t - 1.0
    return label

