import pandas as pd
from sklearn.preprocessing import StandardScaler

def inspect_missing_and_summary(df: pd.DataFrame):
    """
    Check for missing values and basic summary stats on the original DataFrame.
    Returns two strings:
    1) missing_info: counts of missing values per column
    2) summary_stats: output of df.describe()
    """
    # 1) Count missing values in each column
    missing_counts = df.isnull().sum()
    missing_info = missing_counts.to_string()

    # 2) Summary statistics for numeric columns
    summary = df.describe().round(2)
    summary_stats = summary.to_string()

    return missing_info, summary_stats


def normalize_and_encode(df: pd.DataFrame):
    """
    Perform normalization and encoding on the DataFrame:
    - Drop any rows with missing values (if present).
    - One-hot encode 'Location'.
    - Standardize 'Size' and 'Number of Rooms' using StandardScaler.
    Returns:
    - df_preprocessed: the new DataFrame after transformations
    - X: features DataFrame (all columns except 'Price')
    - y: Series of target variable ('Price')
    """
    # 1) Drop missing values (simple strategy)
    df_clean = df.dropna().reset_index(drop=True)

    # 2) One-hot encode 'Location' (drop first to avoid dummy-trap)
    if 'Location' in df_clean.columns:
        df_encoded = pd.get_dummies(df_clean, columns=['Location'], drop_first=True)
    else:
        df_encoded = df_clean.copy()

    # 3) Standardize numerical columns
    numeric_cols = ['Size', 'Number of Rooms']
    scaler = StandardScaler()
    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

    # 4) Separate features (X) and target (y)
    X = df_encoded.drop('Price', axis=1)
    y = df_encoded['Price']

    return df_encoded, X, y
