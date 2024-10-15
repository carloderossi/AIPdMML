import statsmodels.api as sm

import matplotlib.pyplot as plt

from statsmodels.iolib.summary2 import summary_col

def evaluate_ols(X,y):
    print()
    print()
# https://risk-engineering.org/notebook/regression-CCPP.html
# Ordinary Least Squares (OLS) is a method used in linear regression to estimate the parameters of the model. 
# The OLS method aims to find the best-fitting line by minimizing the sum of the squared differences 
# between the observed values and the values predicted by the linear model.
# Add a constant to the features (intercept)
    X = sm.add_constant(X)

    # Fit the OLS model
    model = sm.OLS(y, X).fit()

    # Print the summary
    summary = model.summary()
    print(summary)
    
# ##### Residual plot
    # Fitted values
    fitted_values = model.fittedvalues

    # Residuals
    residuals = model.resid

    plt.figure(figsize=(10, 6))
    plt.scatter(fitted_values, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    manager = plt.get_current_fig_manager()
    manager.set_window_title("OLS model - Residuals")
    plt.show()
    
    """ 
#### Q-Q Plot
    sm.qqplot(residuals, line='45')
    plt.title('Q-Q Plot')
    manager = plt.get_current_fig_manager()
    manager.set_window_title("OLS model - Q-Q Plot")
    plt.show() 
    """

#### Fitted vs. Actual Values Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y, fitted_values)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Fitted Values')
    plt.title('Fitted vs. Actual Values')
    manager = plt.get_current_fig_manager()
    manager.set_window_title("OLS model - Fitted vs. Actual Values")
    plt.show()

##### Summary Table Plot
    df = summary_col([model], stars=True)
    print(df)
