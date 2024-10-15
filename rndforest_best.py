from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

import model_utils as mu

def find_best_params(X_train, y_train, X_test, y_test):
    
# Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    # Initialize the model
    rf = RandomForestRegressor(random_state=42)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")

    # Train the final model with the best parameters
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)

    # Evaluate the final model
    mse, rmse, r2 = mu.evaluate_model(y_test, y_pred)

    print(f"Optimized Random Forest Mean Squared Error: {mse:.2f}")
    print(f"Optimized Random Forest Root Mean Squared Error: {rmse:.2f}")
    print(f"Optimized Random Forest R² Score: {r2:.2f}")



""" 
Best parameters: {'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
Optimized Random Forest Mean Squared Error: 11.48
Optimized Random Forest Root Mean Squared Error: 3.39
Optimized Random Forest R² Score: 0.96 
"""