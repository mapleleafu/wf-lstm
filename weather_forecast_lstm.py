import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

plot_dir = "model_plots"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Function to create an LSTM model
def create_model(input_shape, output_units):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
    model.add(Dropout(0.2))
    model.add(Dense(output_units, activation='relu'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to preprocess the data
def preprocess_data(file_path, seq_length=12):
    df = pd.read_csv(file_path)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.iloc[:, 2:])

    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:(i + seq_length)])
        y.append(scaled_data[i + seq_length])
    X, y = np.array(X), np.array(y)

    train_size = int(len(X) * 0.8)
    val_size = int(len(X) * 0.1)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, df.columns[2:]

# Function to evaluate predictions
def evaluate_predictions(predictions, actuals, feature_names):
    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Plot and save each feature comparison
    for i, feature_name in enumerate(feature_names):
        plt.figure(figsize=(10, 6))
        plt.plot(actuals[:, i], label='Actual', marker='o')
        plt.plot(predictions[:, i], label='Predicted', marker='x')
        plt.title(f'{feature_name} Prediction vs Actual')
        plt.xlabel('Samples')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'{feature_name}_prediction_vs_actual.png'))
        plt.close()

# Function to plot training history
def plot_training_history(history, plot_dir, metric='loss'):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history[metric], label='Train')
    plt.plot(history.history[f'val_{metric}'], label='Validation')
    plt.title(f'Model {metric.title()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric.title())
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f'model_{metric}_plot.png'))
    plt.close()

# Example usage for a single dataset
file_path = 'data_tables/processed_EDIRNE-17050Station.csv'
X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_names = preprocess_data(file_path)

model = create_model(X_train.shape[1:], y_train.shape[1])
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)

# Plot training history
plot_training_history(history, plot_dir, 'loss')

# Predictions and evaluation
predictions = model.predict(X_test)
predictions_inverse = scaler.inverse_transform(predictions)
y_test_inverse = scaler.inverse_transform(y_test)
evaluate_predictions(predictions_inverse, y_test_inverse, feature_names)
