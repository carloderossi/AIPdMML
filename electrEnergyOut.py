import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold

import data_utils as du
import model_utils as mu
import ols_utils as ols

# Load the data
data = du.load_data()
""" 
The columns in the data consist of hourly average ambient variables:
 - Temperature (T) in the range 1.81°C to 37.11°C,
 - Ambient Pressure (AP) in the range 992.89-1033.30 milibar,
 - Relative Humidity (RH) in the range 25.56% to 100.16%
 - Exhaust Vacuum (V) in the range 25.36-81.56 cm Hg
 - Net hourly electrical energy output (PE) 420.26-495.76 MW (Target we are trying to predict) 
 """

# add heat-index as a new feature
# see National Oceanic and Atmospheric Administration (NOAA) and its Weather Prediction Center (www.wpc.ncep.noaa.gov)
data['HI'] = du.calculate_heat_index(data['AT'], data['RH'])      

#first 5 rows of the dataset with the new feature
print(data.head(5))

# Print data analysis
#du.analyse_dataset(data)

print()
data = du.clean_data(data)

# Select features to normalize
features = ['AT', 'AP', 'RH', 'V', 'HI']
data = du.normalize_data(data, features)

print(data.describe())

# Split data between train and evaluate set
X = data[['AT', 'AP', 'RH', 'V', 'HI']]
y = data['PE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=55)

print()

############## TRAIN and EVALUATE LINEAR REGRESSION MODEL ##############
start_time = time.time()
# Initialize the Linear Regression model
lr_model = LinearRegression()
# Train the model
lr_model.fit(X_train, y_train)
train_time_lr = time.time() - start_time

# Make predictions on the test set
start_time = time.time()
y_pred = lr_model.predict(X_test)
pred_time_lr = time.time() - start_time

# Evaluate Model
lr_mean_cv_mse, lr_mean_cv_rmse, lr_mean_cv_r2 = mu.evaluate_model(y_test,y_pred)

print(f"Linear Regression Elapsed Training Time: {train_time_lr}")
print(f"Linear Regression Measured Prediction Time: {pred_time_lr}")

print(f"Linear Regression Cross-Validation MSE: {lr_mean_cv_mse:.2f}")
print(f"Linear Regression Cross-Validation RMSE: {lr_mean_cv_rmse:.2f}")
print(f"Linear Regression Cross-Validation R²: {lr_mean_cv_r2:.2f}")
mu.chart_predictions(y_test,y_pred, 'Linear Regression - Actual vs. Predicted Values with Error Bars (Residuals)')

#mu.explain_model(X, X_train, X_test, lr_model)

# Calculate Cross-validation
""" kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_lr = cross_val_score(lr_model, X, y, cv=kf, scoring='neg_mean_squared_error')
mse_lr = -cv_scores_lr.mean()
rmse_lr = np.sqrt(mse_lr)
print(f"Linear Regression Cross-Validation MSE: {mse_lr:.2f}")
print(f"Linear Regression Cross-Validation RMSE: {rmse_lr:.2f}") """


print()

############## TRAIN and EVALUATE RANDOM FOREST MODEL ##############
start_time = time.time()
# Initialize the Random Forest model
rf_model = RandomForestRegressor()  #(n_estimators=100, random_state=55)
# Train the model
rf_model.fit(X_train, y_train)
train_time_rf = time.time() - start_time

# Make predictions on the test set
start_time = time.time()
y_pred = rf_model.predict(X_test)
pred_time_rf = time.time() - start_time

# Evaluate Model
rf_mean_cv_mse, rf_mean_cv_rmse, rf_mean_cv_r2 = mu.evaluate_model(y_test, y_pred)

print(f"RandomForest Regression Elapsed Training Time: {train_time_rf}")
print(f"RandomForest Regression Measured Prediction Time: {pred_time_rf}")

print(f"RandomForest Regression Cross-Validation MSE: {rf_mean_cv_mse:.2f}")
print(f"RandomForest Regression Cross-Validation RMSE: {rf_mean_cv_rmse:.2f}")
print(f"RandomForest Regression Cross-Validation R²: {rf_mean_cv_r2:.2f}")
mu.chart_predictions(y_test,y_pred, 'RandomForest Regression - Actual vs. Predicted Values with Error Bars (Residuals)')

# Calculate Cross-validation
""" kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_rf = cross_val_score(rf_model, X, y, cv=kf, scoring='neg_mean_squared_error')
mse_rf = -cv_scores_rf.mean()
rmse_rf = np.sqrt(mse_rf)
print(f"Random Forest Cross-Validation MSE: {mse_rf:.2f}")
print(f"Random Forest Cross-Validation RMSE: {rmse_rf:.2f}") """

##### Compare LinearRegression and RandmForest models ######
mu.visualize_results(lr_mean_cv_mse, lr_mean_cv_rmse, lr_mean_cv_r2, rf_mean_cv_mse, rf_mean_cv_rmse, rf_mean_cv_r2)

ols.evaluate_ols(X,y)