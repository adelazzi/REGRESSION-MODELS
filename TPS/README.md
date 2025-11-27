# Concrete Compressive Strength Regression Project

## Overview

This project builds regression models to predict the compressive strength of concrete based on its mixture components and age. The goal is to explore, preprocess, and model the data using various machine learning regression algorithms.

The data includes 1030 samples with 8 features reflecting the composition and age of concrete, while the target is the compressive strength measured in megapascals (MPa).

## Project Structure

- **load_data**: Load the dataset CSV and clean up column names.
- **explore_data**: Perform exploratory data analysis including summary statistics and visualizing the target variable.
- **preprocess_features**: Create additional features like water-to-cement ratio and isolate features and target variables.
- **split_data**: Split the dataset into training and testing sets.
- **build_models**: Initialize several regression models including Linear Regression, SVR, Random Forest, Gradient Boosting, and AdaBoost.
- **evaluate_model**: Train models and evaluate them with MSE, RMSE, and R-squared metrics.
- **plot_feature_importance**: Visualize feature importance for tree-based models.
- **main**: Execute the workflow tying all steps together, identify the best model, and show feature importance.

## Environment Setup and Running the Code

### 1. Create and Activate Python Virtual Environment

python -m venv venv

### Activate the environment:
On Windows

venv\Scripts\activate

### 2. Install Required Packages

pip install pandas numpy scikit-learn matplotlib seaborn


### 3. Prepare Your Dataset

- Place your dataset CSV file (e.g., `Concrete_Data.csv`) in your project directory.
- Ensure the dataset columns match those expected by the code (stripped of trailing spaces).

### 4. Run the Script

python tp3.py


This will load and preprocess the data, train multiple regression models, evaluate their performance, and display visualizations to aid interpretation.

## Notes

- The script automatically strips trailing whitespaces from column names to prevent common errors.
- Scaling is applied only to models sensitive to feature scaling (Linear Regression, SVR).
- Feature engineering includes the water-to-cement ratio, a significant predictor in concrete strength.
- The best model is selected based on minimizing the root mean squared error (RMSE).

---

This documentation provides a clear workflow from environment setup to model evaluation for the concrete compressive strength regression task.

