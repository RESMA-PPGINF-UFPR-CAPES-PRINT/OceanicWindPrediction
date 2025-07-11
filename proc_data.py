import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import timeseries_dataset_from_array
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dropout
import numpy as np
import xarray as xr
import pandas as pd
import os

os.makedirs('ProcData/LSTM', exist_ok=True)
os.makedirs('ProcData/LSTM-RD', exist_ok=True)
os.makedirs('ProcData/LSTM-RDD', exist_ok=True)

# Ensure metrics files do not already exist
metrics_files = ['ProcData/LSTM/lstm_metrics.txt', 'ProcData/LSTM-RD/lstm_rd_metrics.txt', 'ProcData/LSTM-RDD/lstm_rdd_metrics.txt']
for file in metrics_files:
    if os.path.exists(file):
        os.remove(file)

# Load the datasets
ds1 = xr.open_dataset('./cmems_obs-wind_glo_phy_my_l3-metopa-ascat-asc-0.125deg_P1D-i_multi-vars_48.31W-47.44W_26.56S-25.69S_2016-01-01-2021-11-15.nc')
df1 = ds1.to_dataframe()

ds2 = xr.open_dataset('./cmems_obs-wind_glo_phy_my_l3-metopa-ascat-des-0.125deg_P1D-i_multi-vars_48.31W-47.44W_26.56S-25.69S_2016-01-01-2021-11-15.nc')
df2 = ds2.to_dataframe()

# Drop rows with NaN values
df1.dropna(inplace=True)
df1 = df1.query("latitude == -26.1875 and longitude == -47.4375")

df2.dropna(inplace=True)
df2 = df2.query("latitude == -26.1875 and longitude == -47.4375")

df3 = pd.concat([df1,df2]).sort_values(by='time')
df3 = df3[~df3.index.duplicated(keep='first')]
df3.reset_index(inplace=True)
df3.drop("latitude",axis=1,inplace=True)
df3.drop("longitude",axis=1,inplace=True)
df3 = df3.set_index("time")

#Normalize data
mean = df3[:int(len(df3) * 0.7)].mean(axis=0)
df3 -= mean
std = df3[:int(len(df3) * 0.7)].std(axis=0)
df3 /= std

print(mean)
print(std)

data = pd.date_range(start='2016-01-01', end='2021-11-15', freq='D')
data = df3.reindex(data)
data.interpolate(method='ffill', inplace=True)

data.dropna(inplace=True)

plt.rcParams.update({'font.size': 16})

# Parameters
seq_length = 10  # Example sequence length
batch_size = 32

# Split indices
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.2)
test_size = len(data) - train_size - val_size

# Split data
train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

# Create datasets
def create_dataset(data, seq_length, batch_size):
    return timeseries_dataset_from_array(
        data=data[:-seq_length],
        targets=data[seq_length:],
        sequence_length=seq_length,
        batch_size=batch_size
    )

train_dataset = create_dataset(train_data, seq_length, batch_size)
val_dataset = create_dataset(val_data, seq_length, batch_size)
test_dataset = create_dataset(test_data, seq_length, batch_size)

print(f'Length of train_data: {len(train_data)}')
print(f'Length of val_data: {len(val_data)}')
print(f'Length of test_data: {len(test_data)}')

keras.utils.set_random_seed(812)
tf.config.experimental.enable_op_determinism()

#LSTM


for i in range(10):
    # Define the model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(seq_length, 2)))
    model.add(Dense(2))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Define early stopping and model checkpoint callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model with validation
    history = model.fit(train_dataset, epochs=100, validation_data=val_dataset, callbacks=[early_stopping])

    # Find the epoch with the best validation loss
    best_epoch = np.argmin(history.history['val_loss']) + 1
    print(f'Best epoch: {best_epoch}')

    # Evaluate the model on the test dataset using the best epoch's weights
    loss, mae = model.evaluate(test_dataset)
    print(f'Test Loss {i}: {loss}, Test MAE {i}: {mae*std.mean()}')

    # Store the metrics for the best epoch
    with open(f'ProcData/LSTM/lstm_metrics.txt', 'a') as f:
        denormalized_loss = loss * std.mean() * std.mean()
        denormalized_mae = mae * std.mean()
        f.write(f'Execution {i}: Best Epoch: {best_epoch}, Test Loss: {loss}, Test MAE: {mae}, Denormalized Loss: {denormalized_loss}, Denormalized MAE: {denormalized_mae}\n')

    loss = np.array(history.history["mae"]) * std.mean()
    val_loss = np.array(history.history["val_mae"]) * std.mean()
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Train MAE")
    plt.plot(epochs, val_loss, "b", label="Validation MAE")
    plt.title("Train and Validation MAE")
    plt.axvline(best_epoch, linestyle='--', color='r', label='Best Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    #plt.show()
    plt.savefig(f'ProcData/LSTM/lstm{i}_mae_proc.pdf', format="pdf", bbox_inches="tight")

    # Make predictions
    predictions = model.predict(test_dataset)

    # Extract true values from test dataset
    true_values = []
    for _, targets in test_dataset:
        true_values.extend(targets.numpy())

    # Convert to numpy arrays for easier comparison
    true_values = np.array(true_values)

    # Denormalize the predicted values
    predictions = predictions * np.array(std) + np.array(mean)

    # Denormalize the true values
    true_values = true_values * np.array(std) + np.array(mean)

    # Create subplots for Eastward and Northward wind
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot Eastward Wind
    axes[0].plot(range(len(true_values)), true_values[:, 0], label='Actual Eastward Wind', color='blue')
    axes[0].plot(range(len(predictions)), predictions[:, 0], label='Predicted Eastward Wind', linestyle='dashed', color='red')
    axes[0].set_ylabel('Wind Speed (m/s)')
    axes[0].set_title('Eastward Wind: Actual vs Predicted')
    axes[0].legend(loc='upper right')

    # Plot Northward Wind
    axes[1].plot(range(len(true_values)), true_values[:, 1], label='Actual Northward Wind', color='green')
    axes[1].plot(range(len(predictions)), predictions[:, 1], label='Predicted Northward Wind', linestyle='dashed', color='orange')
    axes[1].set_xlabel('Time (Days)')
    axes[1].set_ylabel('Wind Speed (m/s)')
    axes[1].set_title('Northward Wind: Actual vs Predicted')
    axes[1].legend(loc='upper right')

    # Save the figure
    plt.tight_layout()
    plt.savefig(f'ProcData/LSTM/lstm{i}_pred_proc.pdf', format="pdf", bbox_inches="tight")
    
    # Save the model
    model.save(f'ProcData/LSTM/lstm_model_{i}.keras')

# Calculate mean and std of the metrics
metrics = []
denormalized_metrics = []
with open('ProcData/LSTM/lstm_metrics.txt', 'r') as f:
    for line in f:
        if line.startswith('Execution'):
            parts = line.strip().split(',')
            loss = float(parts[1].split(':')[1].strip())
            mae = float(parts[2].split(':')[1].strip())
            denormalized_loss = float(parts[3].split(':')[1].strip())
            denormalized_mae = float(parts[4].split(':')[1].strip())
            metrics.append((loss, mae))
            denormalized_metrics.append((denormalized_loss, denormalized_mae))

losses, maes = zip(*metrics)
denorm_losses, denorm_maes = zip(*denormalized_metrics)

mean_loss = np.mean(losses)
std_loss = np.std(losses)
mean_mae = np.mean(maes)
std_mae = np.std(maes)

mean_denorm_loss = np.mean(denorm_losses)
mean_denorm_mae = np.mean(denorm_maes)

# Find the best execution regarding the mae
best_execution_index = np.argmin(maes)
best_loss = losses[best_execution_index]
best_mae = maes[best_execution_index]
best_denorm_loss = denorm_losses[best_execution_index]
best_denorm_mae = denorm_maes[best_execution_index]

# Store the mean, std, and denormalized metrics in the file
with open('ProcData/LSTM/lstm_metrics.txt', 'a') as f:
    f.write(f'\nMean Loss: {mean_loss}, Std Loss: {std_loss}\n')
    f.write(f'Mean MAE: {mean_mae}, Std MAE: {std_mae}\n')
    f.write(f'Mean Denormalized Loss: {mean_denorm_loss}, Mean Denormalized MAE: {mean_denorm_mae}\n')
    f.write(f'Std Denormalized Loss: {np.std(denorm_losses)}, Std Denormalized MAE: {np.std(denorm_maes)}\n')
    f.write(f'Best Execution: {best_execution_index}, Best Loss: {best_loss}, Best MAE: {best_mae}\n')
    f.write(f'Best Denormalized Loss: {best_denorm_loss}, Best Denormalized MAE: {best_denorm_mae}\n')

#LSTM-RD
keras.utils.set_random_seed(812)
tf.config.experimental.enable_op_determinism()

for i in range(10):
    # Define the model
    model = Sequential()
    model.add(LSTM(50, recurrent_dropout=0.25, activation='relu', input_shape=(seq_length, 2)))
    model.add(Dense(2))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Define early stopping and model checkpoint callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model with validation
    history = model.fit(train_dataset, epochs=100, validation_data=val_dataset, callbacks=[early_stopping])

    # Find the epoch with the best validation loss
    best_epoch = np.argmin(history.history['val_loss']) + 1
    print(f'Best epoch: {best_epoch}')

    # Evaluate the model on the test dataset using the best epoch's weights
    loss, mae = model.evaluate(test_dataset)
    print(f'Test Loss {i}: {loss}, Test MAE {i}: {mae*std.mean()}')

    # Store the metrics for the best epoch
    with open(f'ProcData/LSTM-RD/lstm_rd_metrics.txt', 'a') as f:
        denormalized_loss = loss * std.mean() * std.mean()
        denormalized_mae = mae * std.mean()
        f.write(f'Execution {i}: Best Epoch: {best_epoch}, Test Loss: {loss}, Test MAE: {mae}, Denormalized Loss: {denormalized_loss}, Denormalized MAE: {denormalized_mae}\n')
    
    loss = np.array(history.history["mae"]) * std.mean()
    val_loss = np.array(history.history["val_mae"]) * std.mean()
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Train MAE")
    plt.plot(epochs, val_loss, "b", label="Validation MAE")
    plt.title("Train and Validation MAE")
    plt.axvline(best_epoch, linestyle='--', color='r', label='Best Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    #plt.show()
    plt.savefig(f'ProcData/LSTM-RD/lstm_rd{i}_mae_proc.pdf', format="pdf", bbox_inches="tight")

    # Make predictions
    predictions = model.predict(test_dataset)

    # Extract true values from test dataset
    true_values = []
    for _, targets in test_dataset:
        true_values.extend(targets.numpy())

    # Convert to numpy arrays for easier comparison
    true_values = np.array(true_values)

    # Denormalize the predicted values
    predictions = predictions * np.array(std) + np.array(mean)

    # Denormalize the true values
    true_values = true_values * np.array(std) + np.array(mean)

    # Create subplots for Eastward and Northward wind
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot Eastward Wind
    axes[0].plot(range(len(true_values)), true_values[:, 0], label='Actual Eastward Wind', color='blue')
    axes[0].plot(range(len(predictions)), predictions[:, 0], label='Predicted Eastward Wind', linestyle='dashed', color='red')
    axes[0].set_ylabel('Wind Speed (m/s)')
    axes[0].set_title('Eastward Wind: Actual vs Predicted')
    axes[0].legend(loc='upper right')

    # Plot Northward Wind
    axes[1].plot(range(len(true_values)), true_values[:, 1], label='Actual Northward Wind', color='green')
    axes[1].plot(range(len(predictions)), predictions[:, 1], label='Predicted Northward Wind', linestyle='dashed', color='orange')
    axes[1].set_xlabel('Time (days)')
    axes[1].set_ylabel('Wind Speed (m/s)')
    axes[1].set_title('Northward Wind: Actual vs Predicted')
    axes[1].legend(loc='upper right')

    # Save the figure
    plt.tight_layout()
    plt.savefig(f'ProcData/LSTM-RD/lstm_rd{i}_pred_proc.pdf', format="pdf", bbox_inches="tight")

    # Save the model
    model.save(f'ProcData/LSTM-RD/lstm_rd_model_{i}.keras')

# Calculate mean and std of the metrics
metrics = []
denormalized_metrics = []
with open('ProcData/LSTM-RD/lstm_rd_metrics.txt', 'r') as f:
    for line in f:
        if line.startswith('Execution'):
            parts = line.strip().split(',')
            loss = float(parts[1].split(':')[1].strip())
            mae = float(parts[2].split(':')[1].strip())
            denormalized_loss = float(parts[3].split(':')[1].strip())
            denormalized_mae = float(parts[4].split(':')[1].strip())
            metrics.append((loss, mae))
            denormalized_metrics.append((denormalized_loss, denormalized_mae))

losses, maes = zip(*metrics)
denorm_losses, denorm_maes = zip(*denormalized_metrics)

mean_loss = np.mean(losses)
std_loss = np.std(losses)
mean_mae = np.mean(maes)
std_mae = np.std(maes)

mean_denorm_loss = np.mean(denorm_losses)
mean_denorm_mae = np.mean(denorm_maes)

# Find the best execution regarding the mae
best_execution_index = np.argmin(maes)
best_loss = losses[best_execution_index]
best_mae = maes[best_execution_index]
best_denorm_loss = denorm_losses[best_execution_index]
best_denorm_mae = denorm_maes[best_execution_index]

# Store the mean, std, and denormalized metrics in the file
with open('ProcData/LSTM-RD/lstm_rd_metrics.txt', 'a') as f:
    f.write(f'\nMean Loss: {mean_loss}, Std Loss: {std_loss}\n')
    f.write(f'Mean MAE: {mean_mae}, Std MAE: {std_mae}\n')
    f.write(f'Mean Denormalized Loss: {mean_denorm_loss}, Mean Denormalized MAE: {mean_denorm_mae}\n')
    f.write(f'Std Denormalized Loss: {np.std(denorm_losses)}, Std Denormalized MAE: {np.std(denorm_maes)}\n')
    f.write(f'Best Execution: {best_execution_index}, Best Loss: {best_loss}, Best MAE: {best_mae}\n')
    f.write(f'Best Denormalized Loss: {best_denorm_loss}, Best Denormalized MAE: {best_denorm_mae}\n')


# LSTM-RDD
keras.utils.set_random_seed(812)
tf.config.experimental.enable_op_determinism()

for i in range(10):
    # Define the model
    model = Sequential()
    model.add(LSTM(50, recurrent_dropout=0.25, activation='relu', input_shape=(seq_length, 2)))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Define early stopping and model checkpoint callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model with validation
    history = model.fit(train_dataset, epochs=100, validation_data=val_dataset, callbacks=[early_stopping])

    # Find the epoch with the best validation loss
    best_epoch = np.argmin(history.history['val_loss']) + 1
    print(f'Best epoch: {best_epoch}')

    # Evaluate the model on the test dataset using the best epoch's weights
    loss, mae = model.evaluate(test_dataset)
    print(f'Test Loss {i}: {loss}, Test MAE {i}: {mae*std.mean()}')

    # Store the metrics for the best epoch
    with open(f'ProcData/LSTM-RDD/lstm_rdd_metrics.txt', 'a') as f:
        denormalized_loss = loss * std.mean() * std.mean()
        denormalized_mae = mae * std.mean()
        f.write(f'Execution {i}: Best Epoch: {best_epoch}, Test Loss: {loss}, Test MAE: {mae}, Denormalized Loss: {denormalized_loss}, Denormalized MAE: {denormalized_mae}\n')

    loss = np.array(history.history["mae"]) * std.mean()
    val_loss = np.array(history.history["val_mae"]) * std.mean()
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Train MAE")
    plt.plot(epochs, val_loss, "b", label="Validation MAE")
    plt.title("Train and Validation MAE")
    plt.axvline(best_epoch, linestyle='--', color='r', label='Best Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    #plt.show()
    plt.savefig(f'ProcData/LSTM-RDD/lstm_rdd{i}_mae_proc.pdf', format="pdf", bbox_inches="tight")

    # Make predictions
    predictions = model.predict(test_dataset)

    # Extract true values from test dataset
    true_values = []
    for _, targets in test_dataset:
        true_values.extend(targets.numpy())

    # Convert to numpy arrays for easier comparison
    true_values = np.array(true_values)

    # Denormalize the predicted values
    predictions = predictions * np.array(std) + np.array(mean)

    # Denormalize the true values
    true_values = true_values * np.array(std) + np.array(mean)

    # Create subplots for Eastward and Northward wind
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot Eastward Wind
    axes[0].plot(range(len(true_values)), true_values[:, 0], label='Actual Eastward Wind', color='blue')
    axes[0].plot(range(len(predictions)), predictions[:, 0], label='Predicted Eastward Wind', linestyle='dashed', color='red')
    axes[0].set_ylabel('Wind Speed (m/s)')
    axes[0].set_title('Eastward Wind: Actual vs Predicted')
    axes[0].legend(loc='upper right')

    # Plot Northward Wind
    axes[1].plot(range(len(true_values)), true_values[:, 1], label='Actual Northward Wind', color='green')
    axes[1].plot(range(len(predictions)), predictions[:, 1], label='Predicted Northward Wind', linestyle='dashed', color='orange')
    axes[1].set_xlabel('Time (days)')
    axes[1].set_ylabel('Wind Speed (m/s)')
    axes[1].set_title('Northward Wind: Actual vs Predicted')
    axes[1].legend(loc='upper right')

    # Save the figure
    plt.tight_layout()
    plt.savefig(f'ProcData/LSTM-RDD/lstm_rdd{i}_pred_proc.pdf', format="pdf", bbox_inches="tight")

    # Save the model
    model.save(f'ProcData/LSTM-RDD/lstm_rdd_model_{i}.keras')

# Calculate mean and std of the metrics
metrics = []
denormalized_metrics = []
with open('ProcData/LSTM-RDD/lstm_rdd_metrics.txt', 'r') as f:
    for line in f:
        if line.startswith('Execution'):
            parts = line.strip().split(',')
            loss = float(parts[1].split(':')[1].strip())
            mae = float(parts[2].split(':')[1].strip())
            denormalized_loss = float(parts[3].split(':')[1].strip())
            denormalized_mae = float(parts[4].split(':')[1].strip())
            metrics.append((loss, mae))
            denormalized_metrics.append((denormalized_loss, denormalized_mae))

losses, maes = zip(*metrics)
denorm_losses, denorm_maes = zip(*denormalized_metrics)

mean_loss = np.mean(losses)
std_loss = np.std(losses)
mean_mae = np.mean(maes)
std_mae = np.std(maes)

mean_denorm_loss = np.mean(denorm_losses)
mean_denorm_mae = np.mean(denorm_maes)

# Find the best execution regarding the mae
best_execution_index = np.argmin(maes)
best_loss = losses[best_execution_index]
best_mae = maes[best_execution_index]
best_denorm_loss = denorm_losses[best_execution_index]
best_denorm_mae = denorm_maes[best_execution_index]

# Store the mean, std, and denormalized metrics in the file
with open('ProcData/LSTM-RDD/lstm_rdd_metrics.txt', 'a') as f:
    f.write(f'\nMean Loss: {mean_loss}, Std Loss: {std_loss}\n')
    f.write(f'Mean MAE: {mean_mae}, Std MAE: {std_mae}\n')
    f.write(f'Mean Denormalized Loss: {mean_denorm_loss}, Mean Denormalized MAE: {mean_denorm_mae}\n')
    f.write(f'Std Denormalized Loss: {np.std(denorm_losses)}, Std Denormalized MAE: {np.std(denorm_maes)}\n')
    f.write(f'Best Execution: {best_execution_index}, Best Loss: {best_loss}, Best MAE: {best_mae}\n')
    f.write(f'Best Denormalized Loss: {best_denorm_loss}, Best Denormalized MAE: {best_denorm_mae}\n')

    
