import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def load_data():
    # Load the data
    # https://en.wikipedia.org/wiki/Combined_cycle_power_plant
    # https://storage.googleapis.com/aipi_datasets/CCPP_data.csv
    data = pd.read_csv('./data/CCPP_data.csv')
    # Print first 5 rows of data
    print('\nFirst 5 rows of data:')
    print(data.head(5))
    print("\nLoaded data:\n", data)
    return data

def analyse_dataset(data):
    #distribution of dataset
    print('\nDistribution of dataset:')
    print(data.describe())

    # Display histograms for each feature
    data.hist(bins=20, figsize=(10, 8))
    # Add a title
    plt.suptitle("Distribution Histograms of Features", fontsize=16)
    manager = plt.get_current_fig_manager()
    manager.set_window_title("Distribution Histograms of Features")
    plt.show()

    # Display scatter plots for each pair of features
    """ pd.plotting.scatter_matrix(data, figsize=(12, 10))
    plt.show() """

    sns.pairplot(data)
    #plt.suptitle("Scatter plots for each pair of Features", fontsize=16)
    # Change the window title
    manager = plt.get_current_fig_manager()
    manager.set_window_title("Scatter plots for each pair of Features")
    plt.show()
    
    """ # 3D plot 
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(projection="3d")
fig.add_axes(ax)
ax.scatter(data["AT"], data["V"], data["PE"])
ax.set_xlabel("AT")
ax.set_ylabel("V")
ax.set_zlabel("PE")
ax.set_facecolor("white")
plt.show()
 """

    # Calculate the correlation matrix
    correlation_matrix = data.corr()

    # Print the correlation matrix
    print('\nData correlation matrix:')
    print(correlation_matrix)

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # Draw the heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)

    # Add titles and labels
    plt.title("Correlation Matrix Heatmap")
    manager = plt.get_current_fig_manager()
    manager.set_window_title("Correlation Matrix - Heatmap")
    plt.show()
    
def clean_data(data):
    # Check for null values
    null_values = data.isnull().sum()
    print("Null values in each column:\n", null_values)
    
    # Optionally fill or drop missing values
    data = data.dropna()

    # Check data types
    print('\nCheck DataTypes')
    print(data.dtypes)

    # Check for invalid values (e.g., values outside specified ranges)
    invalid_temp = data[(data['AT'] < 1.81) | (data['AT'] > 37.11)]
    invalid_ap = data[(data['AP'] < 992.89) | (data['AP'] > 1033.30)]
    invalid_rh = data[(data['RH'] < 25.56) | (data['RH'] > 100.16)]
    invalid_v = data[(data['V'] < 25.36) | (data['V'] > 81.56)]

    print("\nInvalid Temperature values:\n", invalid_temp)
    print("\nInvalid Ambient Pressure values:\n", invalid_ap)
    print("\nInvalid Relative Humidity values:\n", invalid_rh)
    print("\nInvalid Exhaust Vacuum values:\n", invalid_v)

    # Drop rows with null values
    data = data.dropna()

    # Drop rows with invalid values
    data = data[(data['AT'] >= 1.81) & (data['AT'] <= 37.11)]
    data = data[(data['AP'] >= 992.89) & (data['AP'] <= 1033.30)]
    data = data[(data['RH'] >= 25.56) & (data['RH'] <= 100.16)]
    data = data[(data['V'] >= 25.36) & (data['V'] <= 81.56)]

    print()
    print("Data after removing invalid rows:\n", data)
    
    return data

def normalize_data(data, features):
    # Initialize the scaler
    scaler = MinMaxScaler()

    # Apply the scaler to the features
    data[features] = scaler.fit_transform(data[features])

    print()
    print("Normalized data:\n", data)
    
    return data

def calculate_heat_index(T, RH):
    # Coefficients for the heat index formula
    # see National Oceanic and Atmospheric Administration (NOOA) and the Weather Center
    # see https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml#:~:text=HI%20%3D%200.5%20%2A%20%7BT%20%2B%2061.0%20%2B,with%20any%20adjustment%20as%20described%20above%20is%20applied.
    c1 = -42.379
    c2 = 2.04901523
    c3 = 10.14333127
    c4 = -0.22475541
    c5 = -6.83783e-3
    c6 = -5.481717e-2
    c7 = 1.22874e-3
    c8 = 8.5282e-4
    c9 = -1.99e-6

    # Calculate heat index
    HI = (c1 + (c2 * T) + (c3 * RH) + (c4 * T * RH) +
          (c5 * T**2) + (c6 * RH**2) + (c7 * T**2 * RH) +
          (c8 * T * RH**2) + (c9 * T**2 * RH**2))
    
    return HI