import matplotlib.pyplot as plt
import seaborn as sns

def plot_time_series(df, column, title, xlabel='Date', ylabel='Value', legend_label=None):
    """Plots a time series for a specified column."""
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df[column], label=legend_label or column)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_rolling_statistics(df, columns, title, xlabel='Date', ylabel='Value', window_labels=None):
    """Plots rolling statistics for specified columns (e.g., moving averages)."""
    plt.figure(figsize=(14, 7))
    for idx, col in enumerate(columns):
        label = window_labels[idx] if window_labels else col
        plt.plot(df.index, df[col], label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_correlation_matrix(df, title="Correlation Matrix"):
    """Plots the correlation matrix for a DataFrame."""
    corr_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title(title)
    plt.show()

def plot_bar_chart(df, column, title, xlabel='Date', ylabel='Value'):
    """Plots a bar chart for a specified column (e.g., trade volume)."""
    plt.figure(figsize=(14, 7))
    plt.bar(df.index, df[column], label=column)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()
