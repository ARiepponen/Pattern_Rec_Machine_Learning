import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os
from multiprocessing import Process



def class_acc(pred, gt):
    list_len = len(pred)
    correct_guesses = 0
    for i in range(list_len):
        if (pred[i] == gt[i]):
            correct_guesses += 1

    correct_percent = 100 * (correct_guesses/list_len)

    return correct_percent


def unpickle(file):
    with open(file, 'rb') as f:
        dct = pickle.load(f, encoding="latin1")
    return dct


def cifar10_classifier_random(data):
    random_label_list = []

    for value in range(len(data)):
        random_label_list.append(random.randint(0,9))

    return random_label_list

def cifar10_classifier_1nn(X, trdata, trlabels):
    X = X.astype(np.int32)
    trdata = trdata.astype(np.int32)
    
    smallest_error = 0
    
    i = 0
    for row in trdata:
        subs_vector = np.subtract(X,row)
        error_vector = pow(subs_vector,2)
        error = np.sum(error_vector)
        if(i == 0 or (smallest_error > error)):
             smallest_error = error
             error_index = i
        i += 1

    
    return trlabels[error_index]
      

def test_data(X, training_data, training_labels):
    
    
    prediction_data = np.array([])

    start_time = time.time()
    i = 1
    for sample in X:

        #process = Process(target=cifar10_classifier_1nn, args=(sample, training_data, training_labels)) #Modaus
        
        print("Test", i, "/", len(X))


        #prediction_data = np.append(prediction_data, process) #modaus

        #process.start() #Modaus

        prediction_data = np.append(prediction_data, cifar10_classifier_1nn(sample, training_data, training_labels))

        if(i%100 == 0):
            print("Time used:", time.time()-start_time, " seconds")
        i += 1

    return prediction_data




def main():

    start_time = time.time()

    #datadict = unpickle('/home/kamarain/Data/cifar-10-batches-py/data_batch_1')
    #datadict = unpickle('/home/kamarain/Data/cifar-10-batches-py/test_batch')
    testdict = unpickle('C:/Users/Antti/Documents/Koulu/Intro_Pat_rec_Machine/Ex_2/venv/cifar-10-batches-py/test_batch')
    datadict = unpickle('C:/Users/Antti/Documents/Koulu/Intro_Pat_rec_Machine/Ex_2/venv/cifar-10-batches-py/data_batch_1')
    datadict2 = unpickle('C:/Users/Antti/Documents/Koulu/Intro_Pat_rec_Machine/Ex_2/venv/cifar-10-batches-py/data_batch_2')
    datadict3 = unpickle('C:/Users/Antti/Documents/Koulu/Intro_Pat_rec_Machine/Ex_2/venv/cifar-10-batches-py/data_batch_3')
    datadict4 = unpickle('C:/Users/Antti/Documents/Koulu/Intro_Pat_rec_Machine/Ex_2/venv/cifar-10-batches-py/data_batch_4')
    datadict5 = unpickle('C:/Users/Antti/Documents/Koulu/Intro_Pat_rec_Machine/Ex_2/venv/cifar-10-batches-py/data_batch_5')

    X2 = datadict2["data"]
    X3 = datadict3["data"]
    X4 = datadict4["data"]
    X5 = datadict5["data"]

    labels2 = datadict2["labels"]
    labels3 = datadict3["labels"]
    labels4 = datadict4["labels"]
    labels5 = datadict5["labels"]

    X = testdict["data"]
    Y = testdict["labels"]
    
    training_data = datadict["data"]
    training_labels = datadict["labels"]
    
    training_data = np.concatenate((training_data, X2,X3,X4,X5), axis = 0)
    training_labels = np.concatenate((training_labels, labels2, labels3, labels4, labels5), axis = 0)

    print(len(training_data))


    #X = X[1:1000:1]
    #training_data = training_data[1:1000:1]
    #training_labels = training_labels[1:1000:1]
    #gt = Y[1:1000:1]
    #labeldict = unpickle('C:/Users/Antti/Documents/Koulu/Intro_Pat_rec_Machine/Ex_2/venv/cifar-10-batches-py/batches.meta')
    #label_names = labeldict["label_names"]
    #pred = cifar10_classifier_random(X) #For classifier testing



    pred = test_data(X, training_data, training_labels)
    
    gt = Y
    accuracy = class_acc(pred, gt)

    print("Correct guess amount is ", str(accuracy), "%")
    print("Total time for the simulation is ", time.time()-start_time, " seconds")


'''    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    Y = np.array(Y)

    for i in range(X.shape[0]):
        # Show some images randomly
        if random.random() > 0.999:
            plt.figure(1)
            plt.clf()
            plt.imshow(X[i])
            plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
            plt.pause(1)
'''
    

main()


