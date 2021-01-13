#Introduction to pattern recognition and Machine learning Week 1 Exercise

import matplotlib

matplotlib.use('Tkagg')


import matplotlib.pyplot as plt
 
import numpy as np

import time


def my_linfit(x,y):
    n = len(x)
    sum_xy = 0
    sum_xx = 0
    for v in range (0,n):
        sum_xy += (x[v] * y[v])
        
        sum_xx += (x[v] * x[v])
        

    a = (sum_xy*n - sum(x)*sum(y))/(n*sum_xx-(sum(x))**2)
    
    b = (sum(y)*sum_xx - sum_xy * sum(x))/(n*sum_xx - (sum(x))**2)
        
    return a,b


def main():
    

    x = []
    y = []

    plt.clf()
    plt.setp(plt.gca(), autoscale_on=False)
    testi = plt.ginput(-1,0,True,1,2,3)


    for value in testi:
        x_value = value[0]
        y_value = value [1]
        x.append(x_value)
        y.append(y_value)

    a,b = my_linfit(x,y)
    
    plt.plot(x,y,'kx')
    xp = np.arange(-2,5,0.1)
    plt.plot(xp,a*xp+b)
    print(f"My fit: a= {a} and b = {b}")
    
    plt.show()

main()