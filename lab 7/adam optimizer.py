import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import MeanAbsoluteError
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

california_housing = fetch_california_housing()
X = california_housing.data
y = california_housing.target


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=[MeanAbsoluteError()])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

def r2_metric(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    return r2

# Train the model with early stopping and display progress
history = model.fit(X_train_scaled, y_train, epochs=200, validation_split=0.2, callbacks=[early_stopping], verbose=1)

# Calculate R^2 score on validation set
y_val_pred = model.predict(X_val_scaled)
r2 = r2_score(y_val, y_val_pred)
print("R^2 score on validation set:", r2)

# Plot the graph for epochs vs. loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

