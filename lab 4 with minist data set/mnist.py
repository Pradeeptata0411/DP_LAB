# othcls.py or any other appropriate file
from tensorflow.keras import layers, models

class SimpleANN():
    def model_arch(self):
        model = models.Sequential()
        model.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=(28, 28, 1), padding="same"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(128, (5, 5), activation='relu', padding="same"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(256, (5, 5), activation='relu', padding="same"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
        return model
