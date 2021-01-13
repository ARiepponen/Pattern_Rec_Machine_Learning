import pickle
import scipy
from scipy.stats import norm, multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from random import random
from skimage.transform import rescale, resize, downscale_local_mean
import time

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

def cifar10_color(X):

    X_mean = np.zeros((len(X),3))

    counter = 0

    for i in range(X.shape[0]):
        if(counter%100 == 0):
            print((counter), "/", len(X), "1x1 images converted")


        img = X[i]
        img_8x8 = resize(img, (8, 8))        
        img_1x1 = resize(img, (1, 1))        
        r_vals = img_1x1[:,:,0].reshape(1*1)
        g_vals = img_1x1[:,:,1].reshape(1*1)
        b_vals = img_1x1[:,:,2].reshape(1*1)
        mu_r = r_vals.mean()   
        mu_g = g_vals.mean()
        mu_b = b_vals.mean()
        X_mean[i,:] = (mu_r, mu_g, mu_b)

        counter += 1

    return X_mean

'''def cifar10_color(X):

    x = X.reshape(len(X), 3, (-1))

    Xf = np.zeros((len(x), 3))

    for i, pic in enumerate(x):
        Xf[i] = np.mean(pic, axis = 1)

    return Xf'''

def cifar10_2x2_color(X):

    X_mean = np.zeros((len(X), 12))

    counter = 0
    for i in range(X.shape[0]):

        if(counter%100 == 0):
            print((counter), "/", len(X), "2x2 images converted")

        img = X[i]
        img2x2 = resize(img, (2,2))
        
        img1 = img2x2[0][0]
        img2 = img2x2[0][1]
        img3 = img2x2[1][0]
        img4 = img2x2[1][1]
        
        X_mean[i,:] = np.concatenate((img1,img2,img3,img4), axis = 0)
        counter += 1
    return X_mean

def cifar10_4x4_color(X):

    X_mean = np.zeros((len(X), 48))

    counter = 0
    for i in range(X.shape[0]):
        if(counter%100 == 0):
            print((counter), "/", len(X), "4x4 images converted")

        img = X[i]
        img4x4 = resize(img, (4,4))
        
        img1 = img4x4[0][0]
        img2 = img4x4[0][1]
        img3 = img4x4[0][1]

        X_mean[i, :] = np.concatenate((img4x4[0][0],img4x4[0][1],img4x4[0][2],img4x4[0][3],img4x4[1][0],img4x4[1][1],img4x4[1][2],
        img4x4[1][3],img4x4[2][0],img4x4[2][1],img4x4[2][2],img4x4[2][3],img4x4[3][0],img4x4[3][1],img4x4[3][2],img4x4[3][3]), axis = 0)
        
        counter += 1

    return X_mean

def cifar10_8x8_color(X):
    X_mean = np.zeros((len(X), 192))

    counter = 0
    for i in range(X.shape[0]):

        if(counter%100 == 0):

            print((counter), "/", len(X), "8x8 images converted")

        img = X[i]
        
        img8x8 = resize(img, (8,8))
        X_mean[i, :] = img8x8.reshape(1,192)
        
        counter += 1

    return X_mean

def cifar10_16x16_color(X):

    X_mean = np.zeros((len(X), 768))

    counter = 0
    for i in range(X.shape[0]):

        if(counter%100 == 0):

            print((counter), "/", len(X), "16x16 images converted")

        img = X[i]
        
        img16x16 = resize(img, (16,16))
        X_mean[i, :] = img16x16.reshape(1,768)
        
        counter += 1

    return X_mean


def cifar_10_bayes_learn(X, Y):

    mean_matrix = []
    covariance_matrix = []
    priors_matrix = []

    classes = [[] for _ in range(10)]

    for i, item, in enumerate(X):
        classes[Y[i]].append(item)
    classes = np.asarray(classes)
    for i, item in enumerate(classes):
        
        rgb_mean_3val = np.mean(item, axis = 0)
        
        rgb_covar_3val = np.cov(item, rowvar = False)
        
        mean_matrix.append(rgb_mean_3val)
        covariance_matrix.append(rgb_covar_3val)
        priors_matrix.append(0.1)

    mean_matrix = np.asarray(mean_matrix)
    covariance_matrix = np.asarray(covariance_matrix)
    priors_matrix = np.asarray(priors_matrix)  
    

    return mean_matrix, covariance_matrix, priors_matrix

def cifar_10_naivebayes_learn(X, Y):

    mean_matrix = []
    variance_matrix = []
    priors_matrix = []

    classes = [[] for _ in range(10)]

    for i, item, in enumerate(X):
        classes[Y[i]].append(item)
    classes = np.asarray(classes)
    
    for i, item in enumerate(classes):
        
        rgb_mean_3val = np.mean(item, axis = 0)
        
        rgb_var_3val = np.std(item, axis = 0)
        
        mean_matrix.append(rgb_mean_3val)
        variance_matrix.append(rgb_var_3val)
        priors_matrix.append(0.1)
    
    mean_matrix = np.asarray(mean_matrix)
    variance_matrix = np.asarray(variance_matrix)
    priors_matrix = np.asarray(priors_matrix)
    
    return mean_matrix, variance_matrix, priors_matrix


def learn_data_naivebayes(X, mean_matrix, variance_matrix, priors_matrix):
    
    prediction_data = np.array([])
    i = 1
    for sample in X:

        print("Test", i, "/", len(X), " Naivebayes")
        prediction_data = np.append(prediction_data, cifar10_classifier_naivebayes(sample, mean_matrix, variance_matrix, priors_matrix))
        i += 1

    return prediction_data

def learn_data_bayes(X, mean_matrix, covariance_matrix, priors_matrix):

    prediction_data = np.array([])
    i = 1
    for sample in X:
        
        print("Test", i, "/", len(X), " Bayes")
        prediction_data = np.append(prediction_data, cifar10_classifier_bayes(sample, mean_matrix, covariance_matrix, priors_matrix))
        i += 1

    return prediction_data

def cifar10_classifier_bayes(sample, mean_matrix, covariance_matrix, priors_matrix):

    probabilities = np.zeros(10)
    
    for i, item in enumerate(mean_matrix):
        
        prob = scipy.stats.multivariate_normal(item, covariance_matrix[i]).pdf(sample)
        
        #probabilities = np.append(probabilities, prob.pdf(sample))
        
        probabilities[i] = prob

        #probabilities[i] = np.sum(multivariate_normal.logpdf(sample, item[i], covariance_matrix[i]) * priors_matrix[i])

    return np.argmax(probabilities)


def cifar10_classifier_naivebayes(X, mean_matrix, variance_matrix, priors_matrix):
    
    probabilities = np.zeros(10)
    
    for i, item in enumerate(mean_matrix):
        
        probabilities[i] = ((norm.pdf(X[0], item[0], variance_matrix[i][0]) * norm.pdf(X[1], item[1], variance_matrix[i][1]) * norm.pdf(X[2], item[2], variance_matrix[i][2])) * priors_matrix[i])
        #probabilities = np.append(probabilities, sc.multivariate_normal.pdf(X, mean = item[i], cov = variance_matrix[i]))
    
    return np.argmax(probabilities)

def class_acc(pred, gt):

    list_len = len(pred)
    correct_guesses = 0
    for i in range(list_len):
        if (pred[i] == gt[i]):
            correct_guesses += 1

    correct_percent = 100 * (correct_guesses/list_len)

    return correct_percent


def main():


    start_time = time.time()

    testdict = unpickle('C:/Users/Antti/Documents/Koulu/Intro_Pat_rec_Machine/Ex_3/venv/cifar-10-batches-py/test_batch')
    datadict = unpickle('C:/Users/Antti/Documents/Koulu/Intro_Pat_rec_Machine/Ex_3/venv/cifar-10-batches-py/data_batch_1')
    datadict2 = unpickle('C:/Users/Antti/Documents/Koulu/Intro_Pat_rec_Machine/Ex_3/venv/cifar-10-batches-py/data_batch_2')
    datadict3 = unpickle('C:/Users/Antti/Documents/Koulu/Intro_Pat_rec_Machine/Ex_3/venv/cifar-10-batches-py/data_batch_3')
    datadict4 = unpickle('C:/Users/Antti/Documents/Koulu/Intro_Pat_rec_Machine/Ex_3/venv/cifar-10-batches-py/data_batch_4')
    datadict5 = unpickle('C:/Users/Antti/Documents/Koulu/Intro_Pat_rec_Machine/Ex_3/venv/cifar-10-batches-py/data_batch_5')


    X = datadict["data"]
    Y = datadict["labels"]

    X2 = datadict2["data"]
    X3 = datadict3["data"]
    X4 = datadict4["data"]
    X5 = datadict5["data"]
    Y2 = datadict2["labels"]
    Y3 = datadict3["labels"]
    Y4 = datadict4["labels"]
    Y5 = datadict5["labels"]

    X = np.concatenate((X,X2,X3,X4,X5), axis = 0)
    Y = np.concatenate((Y,Y2,Y3,Y4,Y5), axis = 0)

    testX = testdict["data"]
    testY = testdict["labels"]

    labeldict = unpickle('C:/Users/Antti/Documents/Koulu/Intro_Pat_rec_Machine/Ex_3/venv/cifar-10-batches-py/batches.meta')
    #label_names = labeldict["label_names"]

    X = X.reshape(len(X), 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    testX = testX.reshape(len(testX), 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    Y = np.array(Y)

    print(X[0])

    print("Converting images...")

    data1x1_set_X_rgb = cifar10_color(X)
    test1x1_set_X_rgb = cifar10_color(testX)
    #data2x2_set_X_rgb = cifar10_2x2_color(X)
    #test2x2_set_X_rgb = cifar10_2x2_color(testX)
    #data4x4_set_X_rgb = cifar10_4x4_color(X)
    #test4x4_set_X_rgb = cifar10_4x4_color(testX)
    #data8x8_set_X_rgb = cifar10_8x8_color(X)
    #test8x8_set_X_rgb = cifar10_8x8_color(testX)
    #data16x16_set_X_rgb = cifar10_16x16_color(X)
    #test16x16_set_X_rgb = cifar10_16x16_color(testX)

    
    mean_matrix, covariance_matrix, priors_matrix = cifar_10_bayes_learn(data1x1_set_X_rgb, Y)
    pred_bayes = learn_data_bayes(test1x1_set_X_rgb, mean_matrix, covariance_matrix, priors_matrix)
    bayes_accuracy1x1 = class_acc(pred_bayes, testY)

    '''mean_matrix, covariance_matrix, priors_matrix = cifar_10_bayes_learn(data2x2_set_X_rgb, Y)
    pred_bayes = learn_data_bayes(test2x2_set_X_rgb, mean_matrix, covariance_matrix, priors_matrix)
    bayes_accuracy2x2 = class_acc(pred_bayes, testY)

    mean_matrix, covariance_matrix, priors_matrix = cifar_10_bayes_learn(data4x4_set_X_rgb, Y)
    pred_bayes = learn_data_bayes(test4x4_set_X_rgb, mean_matrix, covariance_matrix, priors_matrix)
    bayes_accuracy4x4 = class_acc(pred_bayes, testY)

    mean_matrix, covariance_matrix, priors_matrix = cifar_10_bayes_learn(data8x8_set_X_rgb, Y)
    pred_bayes = learn_data_bayes(test8x8_set_X_rgb, mean_matrix, covariance_matrix, priors_matrix)
    bayes_accuracy8x8 = class_acc(pred_bayes, testY)'''

    #mean_matrix, covariance_matrix, priors_matrix = cifar_10_bayes_learn(data16x16_set_X_rgb, Y)
    #pred_bayes = learn_data_bayes(test16x16_set_X_rgb, mean_matrix, covariance_matrix, priors_matrix)
    #bayes_accuracy16x16 = class_acc(pred_bayes, testY)


    #bayes_start_time = time.time()
    #bayes_time = time.time()-bayes_start_time
    mean_matrix, variance_matrix, priors_matrix = cifar_10_naivebayes_learn(data1x1_set_X_rgb, Y)
    #naivebayes_start_time = time.time()
    pred_naivebayes = learn_data_naivebayes(test1x1_set_X_rgb, mean_matrix, variance_matrix, priors_matrix)
    #naivebayes_time = time.time()-naivebayes_start_time
    naivebayes_accuracy = class_acc(pred_naivebayes, testY)
    print("Correct guess amount for naivebayes accuracy is ", str(naivebayes_accuracy), "%")
    print("Correct guess amount for 1x1 bayes accuracy is ", str(bayes_accuracy1x1), "%")


    '''print("Correct guess amount for 2x2 bayes accuracy is ", str(bayes_accuracy2x2), "%")
    print("Correct guess amount for 4x4 bayes accuracy is ", str(bayes_accuracy4x4), "%")
    print("Correct guess amount for 8x8 bayes accuracy is ", str(bayes_accuracy8x8), "%")
    plt.plot(["1x1", "2x2", "4x4", "8x8"], [bayes_accuracy1x1, bayes_accuracy2x2, bayes_accuracy4x4, bayes_accuracy8x8])
    plt.show()'''

    print("Whole process took ", time.time()-start_time, " seconds")

   
    

    '''# Show some images randomly
    if random() > 0.999:
        plt.figure(1)
        plt.clf()
        plt.imshow(img_8x8)
        plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
        plt.pause(1)'''


main()