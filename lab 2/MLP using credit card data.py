import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

file_path = 'credit_risk_dataset.csv'
data = pd.read_csv(file_path)

X = data.drop('loan_status', axis=1)
y = data['loan_status']

X = pd.get_dummies(X, columns=['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
    keras.layers.Dense(64, input_dim=X_train.shape[1], activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

for layer in model.layers:
    weights, biases = layer.get_weights()
    print(f"Weights for layer {layer.name} before training: {weights}")

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))


for layer in model.layers:
    weights, biases = layer.get_weights()
    print(f"Weights for layer {layer.name} after training: {weights}")
