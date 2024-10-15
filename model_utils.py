from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import numpy as np
import matplotlib.pyplot as plt

# Define the RMSE scorer
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_lrmodel(model, X, y):
    # Perform cross-validation for each metric
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    rmse_scorer = make_scorer(rmse, greater_is_better=False)

    mse_cv_scores = cross_val_score(model, X, y, cv=5, scoring=mse_scorer)
    rmse_cv_scores = cross_val_score(model, X, y, cv=5, scoring=rmse_scorer)

    # Cross-validate to get R² scores
    r2_cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

    # Calculate the mean and standard deviation of the scores
    mean_cv_mse = -mse_cv_scores.mean()
    mean_cv_rmse = -rmse_cv_scores.mean()
    mean_cv_r2 = r2_cv_scores.mean()
    
    return mean_cv_mse, mean_cv_rmse, mean_cv_r2

def evaluate_model(y_test, y_pred):
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    # Calculate R² Score
    r2 = r2_score(y_test, y_pred)
    
    return mse, rmse, r2

"""     print(f"Linear Regression Mean Squared Error: {mse:.2f}")
    print(f"Linear Regression Root Mean Squared Error: {rmse:.2f}")
    print(f"Linear Regression R² Score: {r2:.2f}") """

def visualize_results(mse_lr, rmse_lr, r2_lr, mse_rf, rmse_rf, r2_rf):    
    # Prepare data for visualization
    metrics = ['MSE', 'RMSE', 'R²']
    lr_metrics = [mse_lr, rmse_lr, r2_lr]
    rf_metrics = [mse_rf, rmse_rf, r2_rf]

    # Create bar chart
    x = np.arange(len(metrics))  # Label locations
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, lr_metrics, width, label='Linear Regression')
    bars2 = ax.bar(x + width/2, rf_metrics, width, label='Random Forest')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    #ax.set_title('Comparison of Linear Regression and Random Forest Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Attach a text label above each bar in *bars*, displaying its height.
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate('{}'.format(round(height, 2)),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(bars1)
    autolabel(bars2)

    fig.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.set_window_title("Comparison of Linear Regression and Random Forest Metrics")
    plt.show()
    
def chart_predictions(y_test, y_pred, title):
    # Calculate the errors (residuals)
    errors = y_test - y_pred

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, label='Predicted vs. Actual', color='blue')

    # Plot the line y = x for reference
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction Line')

    # Add error bars
    plt.errorbar(y_test, y_pred, yerr=np.abs(errors), fmt='o', color='blue', alpha=0.5, label='Error')

    # Add labels and title
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    #plt.title('Actual vs. Predicted Values with Error Bars (Residuals)')
    plt.title(title)
    plt.legend()

    # Show the plot
    manager = plt.get_current_fig_manager()
    #manager.set_window_title("Actual vs. Predicted Values with Error Bars (Residuals)")
    manager.set_window_title(title)
    plt.show()
    
    