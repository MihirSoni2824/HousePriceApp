import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt  # for setting titles, etc.

def split_and_train(X, y):
    """
    Split X/y into train (80%) and test (20%), train LinearRegression on training set,
    return: (model, X_test, y_test, y_pred).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred


def compute_metrics(y_true, y_pred):
    """
    Compute R² and RMSE between true and predicted targets.
    Returns (r2, rmse).
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, rmse


def plot_correlation_heatmap(df, ax):
    """
    Given a DataFrame df, compute the correlation matrix and plot as a heatmap
    on the provided Matplotlib Axes 'ax'. We use Seaborn for styling.
    """
    corr = df.corr()
    sns.heatmap(
        corr, annot=True, cmap="Spectral", center=0, ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Correlation Heatmap 📊", fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)


def plot_actual_vs_predicted(y_test, y_pred, ax):
    """
    Plot Actual vs Predicted prices on Axes 'ax':
    - Scatter of (y_test, y_pred)
    - Diagonal line y = x in red for reference.
    """
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='w', s=60)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax.set_xlabel("Actual Price", fontsize=12)
    ax.set_ylabel("Predicted Price", fontsize=12)
    ax.set_title("Actual vs Predicted Prices 📈", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.4)
