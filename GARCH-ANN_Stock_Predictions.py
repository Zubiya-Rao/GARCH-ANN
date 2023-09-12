import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Load your financial return data here
stock_data = pd.read_csv('NYSEEE.csv')
returns = 100 * stock_data['Close'].pct_change().dropna()

# Create sequences for ANN
sequence_length = 10
X, y = [], []

for i in range(len(returns) - sequence_length):
    X.append(returns[i:i+sequence_length].values)  # Use raw returns for input
    y.append(returns[i+sequence_length])

X, y = np.array(X), np.array(y)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a parameter grid for GARCH model
p_values = range(1, 3)  # Order of the autoregressive component
q_values = range(1, 3)  # Order of the moving average component

best_rmse = float('inf')
best_params = None

for p in p_values:
    for q in q_values:
        try:
            # Create and fit the GARCH model
            model = arch_model(returns, vol='Garch', p=p, q=q)
            results = model.fit(disp='off')

            # Calculate RMSE
            volatility_forecast = np.sqrt(results.conditional_volatility)
            rmse = np.sqrt(mean_squared_error(returns, volatility_forecast))

            # Check if the current model is the best so far
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = (p, q)

            print(f"p={p}, q={q} - RMSE: {rmse}")

        except Exception as e:
            print(f"p={p}, q={q} - Error: {e}")

print("Best GARCH parameters:", best_params)
print("Best GARCH RMSE:", best_rmse)

# Now, let's use the best GARCH model to forecast volatility
best_p, best_q = best_params
best_garch_model = arch_model(returns, vol='Garch', p=best_p, q=best_q)
garch_results = best_garch_model.fit(disp='off')
volatility_forecast = np.sqrt(garch_results.conditional_volatility)

# Define a parameter grid for ANN model
ann_param_grid = {
    'hidden_layer_sizes': [(10,), (50,), (100,)],
    'alpha': [0.0001, 0.001, 0.01],
    'activation': ['relu', 'tanh']
}

# Create the MLPRegressor model
ann_model = MLPRegressor(random_state=42)

# Define the best hyperparameters obtained from the grid search for ANN
best_ann_hidden_layer_size = (10,)  # Replace with your best value
best_ann_alpha = 0.0001  # Replace with your best value
best_ann_activation = 'relu'  # Replace with your best value

# Train the best ANN model with the selected hyperparameters
best_ann_model = MLPRegressor(hidden_layer_sizes=best_ann_hidden_layer_size, alpha=best_ann_alpha, activation=best_ann_activation, random_state=42)
best_ann_model.fit(X_train, y_train)

# Predict future returns using the hybrid GARCH-ANN model
combined_predictions = best_ann_model.predict(X_val)

# Calculate RMSE for the combined predictions
combined_rmse = np.sqrt(mean_squared_error(y_val, combined_predictions))
print(f"Combined Model RMSE: {combined_rmse}")

# Plot actual returns
plt.figure(figsize=(10, 4))
plt.plot(y_val, label='Actual Returns', linestyle='-', markersize=3)

# Plot predicted returns
plt.plot(combined_predictions, label='Predicted Returns', linestyle='-')
plt.title('Hybrid GARCH-ANN Actual vs. Predicted Returns for NYSE')
plt.xlabel('Time')
plt.ylabel('Returns')
plt.legend()
plt.show()
