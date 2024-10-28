import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load dataset from a specified file path."""
    return pd.read_csv(file_path)

def save_data(df, file_path):
    """Save DataFrame to a specified file path."""
    df.to_csv(file_path, index=False)

def create_technical_indicators(df):
    """Create technical indicators for the dataset."""
    # Calculate Simple Moving Averages (SMA)
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    
    # Calculate Logarithmic Returns
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate RSI
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Drop rows with NaN values after calculation
    df.dropna(inplace=True)
    
    return df

def prepare_features_and_target(df, target_col='close'):
    """Prepare features and target variable for modeling."""
    X = df[['open', 'high', 'low', 'volume', 'number_of_trades', 'SMA_20', 'SMA_50', 'RSI']]
    y = df[target_col]
    return X, y

def split_data(X, y, test_size=0.2):
    """Split features and target into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=42)
