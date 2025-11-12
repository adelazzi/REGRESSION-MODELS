import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from CSV file and clean column names."""
    data = pd.read_csv(filepath)
    # Strip whitespace from column names to avoid KeyErrors
    data.columns = data.columns.str.strip()
    return data


def explore_data(df: pd.DataFrame) -> None:
    """Perform basic exploratory data analysis."""
    print("Dataset shape:", df.shape)
    print("\nData types and missing values:\n")
    print(df.info())
    print("\nStatistical summary:\n", df.describe())

    print("\nCorrelation matrix:\n", df.corr())

    # Visualize distribution of target variable
    sns.histplot(df['Concrete compressive strength'], kde=True)
    plt.title("Distribution of Concrete Compressive Strength")
    plt.show()


def preprocess_features(df: pd.DataFrame) -> (pd.DataFrame, pd.Series): # type: ignore
    """Preprocess data: add features, separate features and target."""
    # Feature engineering: water to cement ratio
    df['water_to_cement'] = df['Water'] / df['Cement']

    # Define predictors X and target y
    X = df.drop(columns=['Concrete compressive strength'])
    y = df['Concrete compressive strength']

    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42):
    """Split dataset into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_models(random_state=42):
    """Create a dictionary of regression models."""
    models = {
        'Linear Regression': LinearRegression(),
        'Support Vector Regression': SVR(kernel='rbf'),
        'Random Forest': RandomForestRegressor(random_state=random_state),
        'Gradient Boosting': GradientBoostingRegressor(random_state=random_state),
        'AdaBoost': AdaBoostRegressor(random_state=random_state)
    }
    return models


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Train, predict, and evaluate model performance."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mse, rmse, r2, y_pred


def plot_feature_importance(model, feature_names):
    """Plot feature importance for models that support it."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
        plt.title("Feature Importances")
        plt.show()
    else:
        print("Feature importance not available for this model.")


def main():
    filepath = 'Concrete_Data.csv'  # Update to your CSV file path

    # Load and preprocess data
    df = load_data(filepath)
    explore_data(df)
    X, y = preprocess_features(df)

    # Split dataset
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Scale features (important for Linear Regression and SVR)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models
    models = build_models()

    # Evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nTraining and evaluating: {name}")
        if name in ['Support Vector Regression', 'Linear Regression']:
            mse, rmse, r2, y_pred = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
        else:
            mse, rmse, r2, y_pred = evaluate_model(model, X_train, y_train, X_test, y_test)

        print(f"{name} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        results[name] = {'model': model, 'mse': mse, 'rmse': rmse, 'r2': r2}

    # Select best model based on RMSE
    best_model_name = min(results, key=lambda k: results[k]['rmse'])
    best_model = results[best_model_name]['model']
    print(f"\nBest model: {best_model_name} with RMSE = {results[best_model_name]['rmse']:.4f}")

    # Plot feature importance if available
    print("\nFeature importance for best model:")
    plot_feature_importance(best_model, X.columns)


if __name__ == '__main__':
    main()
