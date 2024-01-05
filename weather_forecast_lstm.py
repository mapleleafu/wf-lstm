import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l1_l2
from keras_tuner import RandomSearch

# Load data
df = pd.read_csv('data_tables/processed_EDIRNE-17050Station.csv')

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Prepare sequences for LSTM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)

seq_length = 12  # Number of months
X, y = create_sequences(scaled_data, seq_length)

# Split the data
train_size = int(len(X) * 0.8)
val_size = int(len(X) * 0.1)
X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

# Convert data to float32
X_train = np.asarray(X_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
X_val = np.asarray(X_val).astype(np.float32)
y_val = np.asarray(y_val).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)

# Define the model-building function for Keras Tuner
def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
                   activation='relu',
                   input_shape=(seq_length, X_train.shape[2])))
    model.add(Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, default=0.25, step=0.05)))
    model.add(Dense(y_train.shape[1], activation='relu'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Run Keras Tuner
tuner = RandomSearch(
    build_model,
    objective='loss',
    max_trials=10,
    executions_per_trial=1,
    directory='my_dir',
    project_name='lstm_weather_forecast'
)

tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Train the best model further
history = best_model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))

# Save the best model
best_model.save('best_lstm_model.keras')

# Evaluate the best model
best_model.evaluate(X_test, y_test)

# Forecasting
predictions = best_model.predict(X_test)

# Iterate over each feature in the dataset
for i in range(y_test.shape[1]):
    if i < 2: # Skip month and year columns
        continue
    plt.figure(figsize=(12, 4))  # Create a new figure for each plot
    plt.plot(y_test[:, i], label=f'Actual {df.columns[i]}')
    plt.plot(predictions[:, i], label=f'Predicted {df.columns[i]}')
    plt.title(f'Prediction of {df.columns[i]}')
    plt.xlabel('Time Steps')
    plt.ylabel(df.columns[i])
    plt.legend()
    # Save the plot as an image file
    plt.savefig(f'plot_{df.columns[i]}.png')

    # plt.show()  # Display the plot