import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import my_lib as ml
data = pd.read_csv('Housing.csv')
x_data = data['area'].values
y_data = data['price'].values
# Normalize data
x_data = (x_data - np.mean(x_data)) / np.std(x_data)
y_data = (y_data - np.mean(y_data)) / np.std(y_data)

x_data = tf.constant(x_data, dtype=tf.float32)
y_data = tf.constant(y_data, dtype=tf.float32)

model = ml.LinearRegression()

epochs = 100
learning_rate = 0.001
clip_value = 1.0
for epoch in range(epochs):
    loss = ml.train_step(model, x_data, y_data, learning_rate, clip_value)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.numpy()}, m: {model.m.numpy()}, c: {model.c.numpy()}')

print("m:", model.m.numpy())
print("c:", model.c.numpy())

ml.plot_lr_graph(model, x_data, y_data)
