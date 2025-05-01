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

ds1 = xr.open_dataset('./cmems_obs-wind_glo_phy_my_l3-metopa-ascat-asc-0.125deg_P1D-i_multi-vars_48.31W-47.44W_26.56S-25.69S_2016-01-01-2021-11-15.nc')
df1 = ds1.to_dataframe()

ds2 = xr.open_dataset('./cmems_obs-wind_glo_phy_my_l3-metopa-ascat-des-0.125deg_P1D-i_multi-vars_48.31W-47.44W_26.56S-25.69S_2016-01-01-2021-11-15.nc')
df2 = ds2.to_dataframe()

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

data = pd.date_range(start='2016-01-01', end='2021-11-15', freq='D')
data = df3.reindex(data)

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
for i in range(5):
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
    print(f'Test Loss {i}: {loss}, Test MAE {i}: {mae}')#*std.mean()}')

    loss = np.array(history.history["mae"]) #* std.mean()
    val_loss = np.array(history.history["val_mae"]) #* std.mean()
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, "bo", label="MAE de Treino")
    plt.plot(epochs, val_loss, "b", label="MAE de Validação")
    plt.title("MAE de treino e validação")
    plt.axvline(best_epoch, linestyle='--', color='r', label='Melhor Época')
    plt.xlabel('Época')
    plt.ylabel('MAE')
    plt.legend()
    #plt.show()
    plt.savefig(f'lstm{i}_mae_proc.pdf', format="pdf", bbox_inches="tight")

    # Make predictions
    predictions = model.predict(test_dataset)

    # Extract true values from test dataset
    true_values = []
    for _, targets in test_dataset:
        true_values.extend(targets.numpy())

    # Convert to numpy arrays for easier comparison
    true_values = np.array(true_values)

    # Denormalize the predicted values
    predictions = predictions #* np.array(std) + np.array(mean)

    # Denormalize the true values
    true_values = true_values #* np.array(std) + np.array(mean)

    # Plot the predicted values vs the true values
    plt.figure(figsize=(12, 6))

    # Plot true values
    plt.plot(range(len(true_values)), true_values[:, 0], label='Actual Eastward Wind', color='blue')
    plt.plot(range(len(true_values)), true_values[:, 1], label='Actual Nothward Wind', color='green')

    # Plot predicted values
    plt.plot(range(len(predictions)), predictions[:, 0], label='Predicted Eastward Wind', linestyle='dashed', color='red')
    plt.plot(range(len(predictions)), predictions[:, 1], label='Predicted Northward Wind', linestyle='dashed', color='orange')

    plt.xlabel('Time')
    plt.ylabel('Wind Speed (m/s)')
    plt.title('Actual Values x Predicted Values')
    plt.legend()
    plt.savefig(f'lstm{i}_pred_raw.pdf', format="pdf", bbox_inches="tight")
