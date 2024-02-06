from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

class DeepANN():
    def CUSTOM_MODEL(self, input_shape, num_classes):
        model = Sequential()
        model.add(Dense(2, activation='relu', input_shape=input_shape))
        model.add(Dense(2, activation='relu'))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
        return model
