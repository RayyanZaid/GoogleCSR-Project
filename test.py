import matplotlib.pyplot as plt
import numpy as np
# Hardcoded sample data


# Create a model object (not trained, just a placeholder)
model = None

# Define the plotTimeSeries function
def plotTimeSeries(history, X_test, y_test, model):
    actual_time_series = [4, 3, 1, 2]  # Actual time series data
    predicted_time_series = [
        [0.05, 0.05, 0.1, 0.8, 1.0],
        [0.05, 0.05, 0.1, 0.8, 0.0],
        [0.05, 0.95, 0.1, 0.8, 0.0],
        [0.05, 0.05, 0.9, 0.8, 0.0]
    ]  # Predicted time series data, where each inner list represents probabilities

    actual_time_series = np.array(actual_time_series)
    predicted_time_series = np.array(predicted_time_series)

    # Extract the selected index with the highest probability for each time step
    selected_indices = [np.argmax(probabilities) for probabilities in predicted_time_series]

    # Plot actual vs. predicted time series
    plt.figure(figsize=(10, 6))
    plt.plot(actual_time_series, label='Actual Time Series', color='blue', marker='o')
    plt.plot(selected_indices, label='Predicted Time Series', color='red', marker='x', linestyle='--')
    plt.xlabel('Time Steps')
    plt.ylabel('Selected Index (0-4)')
    plt.legend()
    plt.title('Actual vs. Predicted Time Series')
    plt.show()

# Test the plotTimeSeries function
plotTimeSeries(None, None, None, model)