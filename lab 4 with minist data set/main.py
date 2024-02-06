import tensorflow as tf
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
from mnist import SimpleANN

if __name__ == "__main__":
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
    print("TrainX: ", trainX.shape)
    print("TestX: ", testX.shape)

    # Add channel dimension to the images
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

    for i in range(1, 10):
        plt.subplot(4, 4, i)
        plt.imshow(trainX[i, :, :, 0], cmap='gray')  # Display the first channel of the image
    plt.show()

    AnnModel = SimpleANN()
    model1 = AnnModel.model_arch()
    print(model1.summary())

    history = model1.fit(trainX.astype(np.float32), trainY,
                         epochs=3,
                         steps_per_epoch=10,
                         validation_split=0.33, batch_size=32)

    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    predictions = model1.predict(testX[:4])

    for i in range(4):
        label_index = np.argmax(predictions[i])
        label = labels[label_index]
        print(f"Prediction for sample {i + 1}: {label}")

    # Display the first test image
    plt.imshow(testX[0, :, :, 0], cmap='gray')
    plt.title(f"True Label: {labels[testY[0]]}, Predicted Label: {labels[np.argmax(predictions[0])]}")
    plt.show()