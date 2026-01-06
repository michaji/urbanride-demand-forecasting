import pickle
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


# -------------------------------------------------
# Feature Engineering
# -------------------------------------------------

# A. Create "Rush Hour" Feature
def identify_rush_hour(row):
    if row['workingday'] == 1:
        if row['hr'] in [7, 8, 9, 17, 18, 19]:
            return 1
    return 0


def prepare_features(df):
    df = df.copy()

    # Convert date column
    df['dteday'] = pd.to_datetime(df['dteday'])

    # Create rush hour feature
    df['is_rush_hour'] = df.apply(identify_rush_hour, axis=1)

    # Define target
    y = df['cnt'].values

    # Features to drop (leakage & non-predictive)
    features_to_drop = [
        'instant',
        'dteday',
        'atemp',
        'casual',
        'registered',
        'cnt'
    ]

    X = df.drop(columns=features_to_drop)

    return X.values, y


# -------------------------------------------------
# Time-Based Split
# -------------------------------------------------
def time_based_split(df, train_ratio=0.7, val_ratio=0.15):
    df = df.sort_values('dteday')

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


# -------------------------------------------------
# Main Training Logic
# -------------------------------------------------
def main():
    print("📥 Loading data...")
    df = pd.read_csv("data/hour.csv")

    train_df, val_df, test_df = time_based_split(df)

    print("🧠 Preparing features...")
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)

    print("\n🤖 Training Gradient Boosting model...")
    model = GradientBoostingRegressor(
        n_estimators=264,
        learning_rate=0.0770752934981686,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("\n📊 Evaluating on validation set...")
    val_preds = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    val_mae = mean_absolute_error(y_val, val_preds)

    print(f"Validation RMSE: {val_rmse:.2f}")
    print(f"Validation MAE: {val_mae:.2f}")

    print("\n🧪 Evaluating on test set...")
    test_preds = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_mae = mean_absolute_error(y_test, test_preds)

    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Test MAE: {test_mae:.2f}")

    print("💾 Saving model...")
    with open("model_2.bin", "wb") as f_out:
        pickle.dump(model, f_out)

    print("\n✅ Training complete. Model saved as model.bin")


if __name__ == "__main__":
    main()
