# Car Price Prediction

Sure! Here's a README file for your use-case, including an overview, setup instructions, and usage examples. You can use this for your GitHub repository.

# Car Selling Price Prediction

This project aims to predict the selling price of cars using various regression models. The dataset includes features such as brand, model, vehicle age, kilometers driven, fuel type, transmission type, mileage, engine size, max power, and seller type.

## Overview

The project includes:
- Data preprocessing and feature engineering
- Model training and evaluation using multiple regression models
- Visualization of model performance and error distribution

## Table of Contents

- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Visualizations](#visualizations)
- [Results](#results)
- [License](#license)

## Getting Started

### Dependencies

To install the required packages, use the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

Alternatively, you can manually install the dependencies:

```bash
pip install pandas
pip install matplotlib
pip install numpy
pip install seaborn
pip install scikit-learn
pip install xgboost
pip install yellowbrick
```

### Data Preprocessing

The dataset includes features such as `car_name`, `brand`, `model`, `vehicle_age`, `km_driven`, `seller_type`, `fuel_type`, `transmission_type`, `mileage`, `engine`, `max_power`, `seats`, and `selling_price`. The preprocessing steps include:

- Dropping unnecessary columns (`car_name`, `model`)
- Label encoding categorical features
- Splitting the data into training and testing sets
- Standardizing the features

### Model Training and Evaluation

The following regression models are trained and evaluated:

- K-Nearest Neighbors Regressor
- Support Vector Regressor
- Linear Regression
- LightGBM Regressor
- Lasso Regression
- Ridge Regression

Each model is evaluated using RMSE and R-Squared metrics.

### Visualizations

The project includes the following visualizations:

1. **Correlation Matrix**:
   - A heatmap showing the correlations between different features.

2. **Actual vs. Predicted Values**:
   - A 3x3 subplot layout comparing actual vs. predicted selling prices for each model with a 45-degree reference line.

3. **Error Distribution**:
   - A 2x3 subplot layout showing the error distribution for each model using Yellowbrick's ResidualsPlot.

## Results

The results of the analysis include model performance metrics and visualizations to understand the predictions and errors better. The models are compared using RMSE and R-Squared values.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Usage

### Example Code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from yellowbrick.regressor import ResidualsPlot

# Load and preprocess data
df = pd.read_csv('car_data.csv')
modeling_data = df.drop(['car_name', 'model'], axis=1)
X = modeling_data.drop('selling_price', axis=1)
y = modeling_data['selling_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train and evaluate models
models = {
    "KNN Regressor": KNeighborsRegressor(),
    "SVR": SVR(),
    "Linear Regression": LinearRegression(),
    "LGBM Regressor": LGBMRegressor(),
    "Lasso": Lasso(),
    "Ridge": Ridge()
}

# Plot Actual vs Predicted
fig, axs = plt.subplots(3, 3, figsize=(20, 15))
axs = axs.flatten()
max_price = max(y_test.max(), *[max(pred) for pred in performance_df["Predictions"]])
for i, model_name in enumerate(models.keys()):
    model = models[model_name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    axs[i].scatter(y_test, y_pred, alpha=0.6)
    axs[i].plot([0, max_price], [0, max_price], color='red', linestyle='--', linewidth=2)
    r2 = r2_score(y_test, y_pred)
    axs[i].set_title(f'Actual vs Predicted - {model_name}', fontsize=20)
    axs[i].set_xlabel('Actual Selling Price', fontsize=18)
    axs[i].set_ylabel('Predicted Selling Price', fontsize=18)
    axs[i].tick_params(axis='x', labelsize=14)
    axs[i].tick_params(axis='y', labelsize=14)
    axs[i].text(0.05, 0.95, f'R-Squared: {r2:.2f}', transform=axs[i].transAxes, fontsize=15, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])
plt.tight_layout()
plt.show()

# Plot Error Distribution
fig, axs = plt.subplots(2, 3, figsize=(20, 10))
axs = axs.flatten()
for i, (model_name, model) in enumerate(models.items()):
    visualizer = ResidualsPlot(model, ax=axs[i])
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.finalize()
    axs[i].set_title(f'Error Distribution - {model_name}', fontsize=20)
    axs[i].set_xlabel('Predicted Selling Price', fontsize=18)
    axs[i].set_ylabel('Residuals', fontsize=18)
    axs[i].tick_params(axis='x', labelsize=16)
    axs[i].tick_params(axis='y', labelsize=16)
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])
plt.tight_layout()
plt.show()
```

## Conclusion

This project provides a comprehensive analysis of car selling price prediction using multiple regression models. The visualizations and metrics help in understanding the performance and accuracy of each model, making it easier to select the best model for this use case.
```

You can save this content in a `README.md` file and include it in your GitHub repository. This provides a detailed overview and instructions for others to understand and use your project.
