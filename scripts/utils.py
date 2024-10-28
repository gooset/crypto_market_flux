def split_data(df, test_size=0.2):
    """
    Split data into training and testing sets.

    Parameters:
        df (pd.DataFrame): DataFrame to split.
        test_size (float): Proportion of data to use for testing.

    Returns:
        tuple: Training and testing DataFrames.
    """
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=test_size, random_state=42)
    return train, test

def calculate_rsi(df, window=14):
    """Calculates the Relative Strength Index (RSI) for the closing prices."""
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def handle_missing_values(df):
    """Fills or drops missing values in the dataset."""
    df.fillna(method='ffill', inplace=True)  # Forward-fill missing values
    df.dropna(inplace=True)  # Drop remaining NA values
    return df

    import pandas as pd


