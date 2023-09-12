#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load stock data into a pandas DataFrame
data = pd.read_csv('NYSE.csv')
# Calculate returns to make the data stationary
data['Returns'] = 100 * data['Close'].pct_change().dropna()

# Create sequences for your analysis
sequence_length = 10
X, y = [], []

for i in range(len(data) - sequence_length):
    X.append(data['Returns'].iloc[i:i + sequence_length].values)
    y.append(data['Returns'].iloc[i + sequence_length])

X, y = np.array(X), np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data using Min-Max scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an imputer to fill missing values with mean or other strategies
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Define a parameter grid for GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(10,), (50,), (100,)],
    'alpha': [0.0001, 0.001, 0.01],
    'activation': ['relu', 'tanh']
}

# Create the MLPRegressor model
model = MLPRegressor(random_state=42)

# Create GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1, verbose=2)

# Fit the model to find the best hyperparameters
grid_search.fit(X_train, y_train)

# Get the best hyperparameters from the grid search
best_hidden_layer_size = grid_search.best_params_['hidden_layer_sizes']
best_alpha = grid_search.best_params_['alpha']
best_activation = grid_search.best_params_['activation']

print(f"Best Hidden Layer Size: {best_hidden_layer_size}, Best Alpha: {best_alpha}, Best Activation: {best_activation}")

# Train the best model with the selected hyperparameters
best_model = MLPRegressor(hidden_layer_sizes=best_hidden_layer_size, alpha=best_alpha, activation=best_activation, random_state=42)
best_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Calculate RMSE on the test set
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {test_rmse}")

# Plotting actual vs. predicted returns
plt.figure(figsize=(10, 4))
plt.plot(y_test, label='Actual Returns')
plt.plot(y_pred, label='Predicted Returns')
plt.xlabel('Time')
plt.ylabel('Returns')
plt.title('ANN Actual vs. Predicted Returns for NYSE')
plt.legend()
plt.show()

# Create a DataFrame to store actual and predicted returns
results_df = pd.DataFrame({'Actual Returns': y_test, 'Predicted Returns': y_pred})

# Display the DataFrame
print(results_df)
