import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
import pickle
import matplotlib.pyplot as plot
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras

from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

def plot_test():

    t = np.arange(0,10,0.1);
    y = np.sin(t)
    plot.plot(t,y)
    plot.title('Training data for regression')
    plot.grid(True, which = 'both')
    plot.show()


def main():

    testdict = unpickle('C:/Users/Antti/Documents/Koulu/Intro_Pat_rec_Machine/Ex_3/venv/cifar-10-batches-py/test_batch')
    datadict = unpickle('C:/Users/Antti/Documents/Koulu/Intro_Pat_rec_Machine/Ex_3/venv/cifar-10-batches-py/data_batch_1')
    datadict2 = unpickle('C:/Users/Antti/Documents/Koulu/Intro_Pat_rec_Machine/Ex_3/venv/cifar-10-batches-py/data_batch_2')
    datadict3 = unpickle('C:/Users/Antti/Documents/Koulu/Intro_Pat_rec_Machine/Ex_3/venv/cifar-10-batches-py/data_batch_3')
    datadict4 = unpickle('C:/Users/Antti/Documents/Koulu/Intro_Pat_rec_Machine/Ex_3/venv/cifar-10-batches-py/data_batch_4')
    datadict5 = unpickle('C:/Users/Antti/Documents/Koulu/Intro_Pat_rec_Machine/Ex_3/venv/cifar-10-batches-py/data_batch_5')


    X1 = datadict["data"]
    Y1 = datadict["labels"]

    X2 = datadict2["data"]
    X3 = datadict3["data"]
    X4 = datadict4["data"]
    X5 = datadict5["data"]
    Y2 = datadict2["labels"]
    Y3 = datadict3["labels"]
    Y4 = datadict4["labels"]
    Y5 = datadict5["labels"]

    testX = testdict["data"]
    testY = testdict["labels"]


    X1 = np.array(X1)
    Y1 = np.array(Y1)
    testX = np.array(testX)
    testY = np.array(testY)

    X1 = np.concatenate((X1, X2,X3,X4,X5), axis = 0)
    Y1 = np.concatenate((Y1,Y2,Y3,Y4,Y5), axis = 0)

    X1 = X1 / 255    
    testX = testX / 255

    #testX = testX[:1000]
    #testY = testY[:1000]

    print(X1[0])
    print(testX[0])
    
    testX = testX.reshape(10000, 32, 32, 3)
    X1 = X1.reshape(len(X1), 32, 32, 3)
    print(testX.shape)

    

    model = Sequential()
    """model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape = (32,32,3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))"""

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    #model.add(Dense(128, activation = 'sigmoid'))

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())

    model.add(Dense(128, activation='sigmoid', kernel_initializer='he_uniform'))
    model.add(Dense(10))

    
    #model.add(Dense(1000, input_shape = (32,32,3), activation = 'sigmoid'))
    #model.add(Dense(192, input_shape = (32,32,3), activation = 'sigmoid'))
    #model.add(Dense(16, activation = 'sigmoid'))
    #model.add(Dense(10, input_dim = 3072, activation = 'sigmoid'))
    #model.add(Dense(32, activation = 'sigmoid'))
    #model.add(Dense(64, activation = 'sigmoid'))
    #model.add(Dense(10, activation = 'sigmoid'))
    #model.add(Dense(10, activation = 'sigmoid'))

    keras.optimizers.SGD(lr=0.4)

    model.compile(optimizer = 'sgd', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ["accuracy"])

    model.fit(X1, Y1, epochs = 10)

    _, test_acc = model.evaluate(testX ,  testY, verbose=1)

    print('\nTest accuracy:', test_acc)


    
    '''model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape = (32,32,3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='mse', metrics=['CategoricalAccuracy'])
    print("Nice")
    model.fit(X1, Y1, epochs = 10)

    _, test_acc = model.evaluate(testX ,  testY, verbose=0)

    
    print('\nTest accuracy:', test_acc)'''

    '''model.add(Dense(10, input_dim = 3072, activation = 'sigmoid'))

    model = keras.Sequential([
        keras.layers.Flatten(input_shape = (32, 32)),
        keras.layers.Dense(128, activation = 'relu'),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ['accuracy']
    )'''

    #testY = tf.placeholder(tf.int32, [None])
    #testX = tf.placeholder(tf.int32, [None])
    plot_test()

main()