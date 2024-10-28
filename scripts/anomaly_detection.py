import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np

def detect_anomalies_isolation_forest(df, features, contamination=0.01, random_state=42):
    """
    Detect anomalies in the dataset using the Isolation Forest algorithm.
    
    Parameters:
    - df: DataFrame, the input data.
    - features: list, columns to use for anomaly detection.
    - contamination: float, the proportion of anomalies in the data.
    - random_state: int, for reproducibility.
    
    Returns:
    - DataFrame with an additional 'anomaly' column indicating outliers (1 for anomaly, -1 for normal).
    """
    # Fit the Isolation Forest model
    isolation_forest = IsolationForest(contamination=contamination, random_state=random_state)
    df['anomaly'] = isolation_forest.fit_predict(df[features])
    
    return df

def add_anomaly_column(df, column, threshold):
    """
    Adds a column to the DataFrame that marks rows as anomalies based on a threshold.
    
    Parameters:
    - df: DataFrame, the input data.
    - column: str, the column to use for anomaly detection.
    - threshold: float, the threshold beyond which a value is considered an anomaly.
    
    Returns:
    - DataFrame with an additional 'anomaly_flag' column where anomalies are marked as 1, otherwise 0.
    """
    df['anomaly_flag'] = np.where(df[column] > threshold, 1, 0)
    return df

def visualize_anomalies(df, feature, anomaly_column='anomaly'):
    """
    Visualize anomalies in a time series by plotting the feature and highlighting anomalies.
    
    Parameters:
    - df: DataFrame, the input data.
    - feature: str, the feature to visualize.
    - anomaly_column: str, the column indicating anomalies (default is 'anomaly').
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(14, 7))
    
    # Plot the feature
    plt.plot(df.index, df[feature], label=feature, color='blue', zorder=1)
    
    # Overlay anomalies
    anomalies = df[df[anomaly_column] == 1]
    plt.scatter(anomalies.index, anomalies[feature], color='red', label='Anomalies', zorder=2)
    
    plt.title(f"{feature} with Anomalies")
    plt.xlabel("Date")
    plt.ylabel(f"{feature}")
    plt.legend()
    plt.grid(True)
    plt.show()

def anomaly_summary(df, anomaly_column='anomaly'):
    """
    Provide a summary of the anomalies detected.
    
    Parameters:
    - df: DataFrame, the input data.
    - anomaly_column: str, the column that marks anomalies.
    
    Returns:
    - Summary dict with counts and proportion of anomalies.
    """
    total = len(df)
    anomalies = len(df[df[anomaly_column] == 1])
    normal = total - anomalies
    anomaly_percentage = (anomalies / total) * 100
    
    summary = {
        "total_records": total,
        "anomalies": anomalies,
        "normal": normal,
        "anomaly_percentage": anomaly_percentage
    }
    
    return summary
