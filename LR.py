import numpy as np 
import pandas as pd 
import random
from math import sqrt 


def rmse_metric(actual, predicted):
    sum_error = 0.0
    for y, y_hat in zip(actual, predicted):
        predicted_error = y - y_hat
        sum_error+= (predicted_error ** 2)
        
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)



def compute_ols_coeffs(x,y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = 0
    denominator = 0 

    for i in range(n):
        numerator += (x[i] - x_mean) * (y[i] - y_mean)
        denominator += (x[i] - x_mean) ** 2 

    #calcualte coefficients 
    slope = numerator / denominator
    intercept =  y_mean - slope * x_mean

    return slope, intercept 



def predict(x, w1, w0):
    return w1 * x + w0

x = np.arange(1, 51)
y = x*3+5


y[np.random.radint(0, len(y), size = 10)] += np.random.randint(-5,5)


w1, w0 = compute_ols_coeffs(x,y)
y_hat = predict(x, w1, w0)

print(w1, w0)


def evaluate_ols(y, y_hat):
    mse = np.mean((y-y_hat) ** 2)
    return mse, np.sqrt(mse)

print(evaluate_ols(y, y_hat))




