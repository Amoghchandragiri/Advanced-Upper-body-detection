import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import pickle

# Load the measurement data
try:
    with open('data/measurements.txt', 'r') as file:
        measurements = np.loadtxt(file)
except Exception as e:
    print(f"Error loading measurements: {e}")
    measurements = None

# Load the mediapipe data
try:
    with open('data/mediapipe_data.txt', 'r') as file:
        mediapipe_data = np.loadtxt(file)
except Exception as e:
    print(f"Error loading mediapipe data: {e}")
    mediapipe_data = None

# Check if data is loaded correctly
if measurements is None or mediapipe_data is None or len(measurements) == 0 or len(mediapipe_data) == 0:
    print("One or both of the data files are empty or not loaded properly.")
    exit()

# Assuming measurements are the labels (y) and mediapipe data is the input (X)
X = mediapipe_data
y = measurements

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")


# Standardize the input data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

# Save the mean and scale values
mean_values = scaler.mean_
scale_values = scaler.scale_

with open('scaler_values.pkl', 'wb') as file:
    pickle.dump((mean_values, scale_values), file)

# Build the model
model = keras.Sequential([
    layers.Flatten(input_shape=(4, 1)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(4)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1)

# Evaluate the model
mse = model.evaluate(X_test, y_test, verbose=0)
print("Mean Squared Error:", mse)

# Save the trained model
model.save('trained_model.h5')