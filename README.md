# Prediction of the Net Hourly Electrical Energy Output of a Combined Cycle Power Plant

## Machine Learning Foundation for Product Managers

**Carlo De Rossi, October 2024**

---

## Assignment

In this project, we will build a model to predict the electrical energy output of a Power Plant, which uses a combination of gas turbines, steam turbines, and heat recovery steam generators to generate power.

## Data Overview

We have a set of 9568 hourly average ambient environmental readings from sensors at the power plant which we will use in our model.

The columns in the dataset consist of hourly average ambient variables:
- Ambient Temperature (AT) in the range 1.81°C to 37.11°C
- Ambient Pressure (AP) in the range 992.89-1033.30 millibar
- Relative Humidity (RH) in the range 25.56% to 100.16%
- Exhaust Vacuum (V) in the range 25.36-81.56 cm Hg
- Net hourly electrical energy output (PE) 420.26-495.76 MW (Target we are trying to predict)

## Approach and Output Metric

For predicting the electrical energy output, we need a regression approach, as the target variable (PE) is continuous (numeric).

Output Metrics to evaluate performance:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² (R-squared) Score

Algorithms:
- Linear Regression
- Random Forest Regressor
- [Optional] Gradient Boosting Regressor
- [Optional] Support Vector Regression (SVR)

## Data Analysis

- **Dataset is clean**: No missing data, no invalid data
- **New Feature**: As suggested by the National Oceanic and Atmospheric Administration (NOAA), we add the Heat Index (a derived feature combining temperature and humidity).
  - **Combining Influences**: The Heat Index offers a comprehensive representation of the actual conditions at the power plant.
  - **Capturing Non-linear Relationships**: The Heat Index captures the compounded effect of temperature and humidity in a non-linear way.
  - **Enhancing Model Performance**: Adding this feature can improve prediction accuracy.
  - **Relevance to the Context**: The combined effect of temperature and humidity can significantly impact operations, making it valuable for precise forecasting.

- **Normalization**: Given the data ranges, we need to normalize the data.
  - **Consistency Across Features**: Normalization brings all features to a similar scale.
  - **Improved Performance**: Algorithms converge faster on normalized data.
  - **Equal Weight and Better Accuracy**: Prevents features with larger ranges from dominating the model, ensuring each feature contributes proportionately.

## Diagrams

- Distribution Histograms of Features
- Correlation Matrix HeatMap

## Linear Regression

Starting with Linear Regression has its perks:
- **Clarity and Interpretability**: Results are easy to interpret.
- **Baseline Performance**: Provides a benchmark.
- **Quick and Efficient**: Fast training.
- **Diagnosing Issues**: Insight into basic relationships in your data.

Given the data diagrams, we expect the Linear Regression model to perform well (~80%).

Data are split into train (~70-80%), evaluate, and test datasets, with random assignment.

### Linear Regression - Evaluation
- Elapsed Training Time: 0.004 seconds
- Measured Prediction Time: 0.001 seconds
- Cross-Validation MSE: 19.51
- Cross-Validation RMSE: 4.42
- Cross-Validation R²: 0.93

### Random Forest - Evaluation
- Elapsed Training Time: 4.314 seconds
- Measured Prediction Time: 0.055 seconds
- Cross-Validation MSE: 11.61
- Cross-Validation RMSE: 3.41
- Cross-Validation R²: 0.96

## Models Comparison

The two models show a significant difference in training and prediction time and comparable accuracy and RMSE values.

Only when the small increase in accuracy justifies the significant training and prediction cost would it make sense not to use the Linear Regression model.

For the context of this exercise, we conclude that the Linear Regression model is accurate enough and continue its evaluation.

## Ordinary Least Squares (OLS)

OLS is a method used in linear regression to estimate the parameters of the model. It aims to find the best-fitting line by minimizing the sum of squared differences between the observed values and the values predicted by the linear model.
- **R-squared: 0.933**: The model explains 93.3% of the variance in the dependent variable.
- **Adjusted R-squared: 0.933**: A high Adjusted R-squared value close to the R-squared value signifies that the model is a good fit and isn’t unnecessarily complex.
- **F-statistic: 2.663e+04**: Measures the overall significance of the model. A large F-statistic indicates that the independent variables collectively have a significant impact.
- **Prob (F-statistic): 0.00**: Indicates that there is virtually no chance that the observed F-statistic is due to random chance.

### OLS - Residual Plot

This plot shows the residuals (errors) on the y-axis and the fitted values on the x-axis. Randomly scattered residuals around the horizontal axis suggest unbiased predictions.

### OLS - Fitted vs. Actual Values Plot

This plot compares the predicted values with the actual values. Points close to the 45-degree reference line suggest that the model's predictions are close to the actual values.

## References

- National Oceanic and Atmospheric Administration (NOAA) and the Weather Prediction Center (www.wpc.ncep.noaa.gov)
- Heat Index Equation: [NOAA Heat Index Equation](https://www.noaa.gov)
- GitHub Public Repo: [GitHub Repository](https://github.com/carloderossi/)
- Jupyter Notebook: [Jupyter Notebook](https://github.com/carloderossi/CCP_NOTEBOOK.jpynb)
