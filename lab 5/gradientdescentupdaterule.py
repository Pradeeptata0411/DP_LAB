import tensorflow as tf
import matplotlib.pyplot as plt

class NeuralNetwork(tf.keras.Model):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(NeuralNetwork, self).__init__()
        self.input_layer = tf.keras.layers.Dense(hidden_sizes[0], activation='relu', input_shape=(input_size,))
        self.hidden_layers = [tf.keras.layers.Dense(size, activation='relu') for size in hidden_sizes[1:]]
        self.output_layer = tf.keras.layers.Dense(output_size)

    def call(self, inputs):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

def main():
    nn = NeuralNetwork(input_size=4, hidden_sizes=[2, 2, 2], output_size=2)
    x_train = tf.random.normal([100, 4])
    y_train = tf.random.normal([100, 2])

    nn.compile(optimizer='sgd', loss='mean_squared_error')
    history = {'loss': []}

    for epoch in range(10):
        history['loss'].append(nn.train_on_batch(x_train, y_train))
        print(f"Epoch {epoch + 1}/10, Loss: {history['loss'][-1]}")

    plt.plot(history['loss'])
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

if __name__ == "__main__":
    main()
