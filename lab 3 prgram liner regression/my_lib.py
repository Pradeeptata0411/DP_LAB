#ecel data
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

class LinearRegression(tf.Module):
    def __init__(self):
        self.m = tf.Variable(1.0, dtype=tf.float32, name='M')
        self.c = tf.Variable(0.0, dtype=tf.float32, name='C')

    def __call__(self, x):
        return tf.add(tf.multiply(self.m, x), self.c)

def mean_reduce_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true))

def train_step(model, inputs, targets, learning_rate=0.001, clip_value=1.0):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = mean_reduce_error(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    gradients = [tf.clip_by_value(grad, -clip_value, clip_value) for grad in gradients]
    optimizer = tf.optimizers.SGD(learning_rate)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def plot_lr_graph(model, x_data, y_data):
    plt.scatter(x_data, y_data, label='Data points')
    plt.plot(x_data, model(x_data), color='red', label='Linear Regression')
    plt.xlabel('Area')
    plt.ylabel('Price')
    plt.title('Linear Regression Fit')
    plt.legend()
    plt.show()