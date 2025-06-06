def detect_outliers_iqr(df, column):
    """
    Identify outliers in 'column' using the IQR method:
    - Q1 = 25th percentile, Q3 = 75th percentile, IQR = Q3 - Q1.
    - Outliers: values < Q1 - 1.5*IQR or > Q3 + 1.5*IQR.
    Returns a Boolean mask of outlier rows.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))
    return mask
