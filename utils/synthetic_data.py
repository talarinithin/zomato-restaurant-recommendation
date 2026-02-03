import numpy as np
import pandas as pd


def add_synthetic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds synthetic operational features:
    - success_rate
    - cancellation_rate

    Logic is based on:
    - rating
    - votes
    - online_order availability
    """

    df = df.copy()
    np.random.seed(42)

    # --- Normalize rating (1–5 → 0–1) ---
    df['rating_norm'] = (df['rate'] - 1) / 4

    # --- Normalize votes safely ---
    if df['votes'].max() != df['votes'].min():
        df['votes_norm'] = (
            (df['votes'] - df['votes'].min()) /
            (df['votes'].max() - df['votes'].min())
        )
    else:
        df['votes_norm'] = 0.5

    # --- Success rate calculation ---
    df['success_rate'] = (
        0.5 * df['rating_norm'] +
        0.3 * df['votes_norm'] +
        0.2 * df['online_order']
    )

    # --- Add small controlled noise ---
    df['success_rate'] += np.random.normal(0, 0.05, len(df))

    # --- Clamp between 0 and 1 ---
    df['success_rate'] = df['success_rate'].clip(0, 1)

    # --- Cancellation rate ---
    df['cancellation_rate'] = 1 - df['success_rate']

    # --- Remove helper columns ---
    df.drop(columns=['rating_norm', 'votes_norm'], inplace=True)

    return df
