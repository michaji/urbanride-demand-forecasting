import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_features(
    df,
    scaler=None,
    fit_scaler=False
):
    df = df.copy()

    # ---- Drop leakage columns if present ----
    leakage_cols = ['casual', 'registered']
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

    # ---- Datetime parsing ----
    #df['year'] = df['datetime'].dt.year
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)


    # ---- Create "Rush Hour" Feature ----
    # Logic: If it's a working day AND the hour is 7-9 or 17-19 (5-7 PM)
    def identify_rush_hour(row):
        if row['workingday'] == 1:
            if row['hr'] in [7, 8, 9, 17, 18, 19]:
                return 1
        return 0
    df['is_rush_hour'] = df.apply(identify_rush_hour, axis=1)

    # ---- Drop original datetime ----
    df = df.drop(columns=['dteday'])

    # ---- Separate target if present ----
    y = None
    if 'count' in df.columns:
        y = df['count'].values
        df = df.drop(columns=['count'])

    # ---- Scaling (for linear models) ----
    numerical_cols = df.columns

    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    else:
        df[numerical_cols] = scaler.transform(df[numerical_cols])

    return df.values, y, scaler


#Time_based Split
def time_based_split(df, train_ratio=0.7, val_ratio=0.15):
    df = df.sort_values('datetime')

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


#train_df, val_df, test_df = time_based_split(df)