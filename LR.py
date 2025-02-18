import numpy as np 
import random
from math import sqrt 
import matplotlib.pyplot as plt 

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


y[np.random.randint(0, len(y), size = 10)] += np.random.randint(-5,5)


w1, w0 = compute_ols_coeffs(x,y)
y_hat = predict(x, w1, w0)

print(w1, w0)


def evaluate_ols(y, y_hat):
    mse = np.mean((y-y_hat) ** 2)
    return mse, np.sqrt(mse)

print(evaluate_ols(y, y_hat))




import matplotlib.pyplot as plt

plt.scatter(x, y, label='Observed Value')
plt.plot(x, y_hat, label='Predicted Value', color='red')
plt.xlabel('<--X-Axis-->')
plt.ylabel('<--Y-Axis-->')
plt.legend()
plt.show()


# gradient descent 
def initialise(dim):
    w1 = np.random.rand(dim)
    w0 = np.random.rand()

    return w1, w0


def compute_cost(x, y, y_hat):
    m = len(y)
    cost = (1/2*m) * np.sum(np.square(y_hat - y))
    return cost

def predict_y(x, w1, w0):
    if len(w1) == 1:
        w1 = w1[0]
        return x + w1 + w0 
    return np.dot(x, w1) + w0   


def update_parameters(x,y, y_hat, cost, w0, w1, learning_rate):
    m = len(y)
    db = (np.sum(y_hat - y)) / m
    dw = (np.dot(y_hat - y, x)) / m

    w0_new = w0 - learning_rate * db
    w1_new = w1 - learning_rate * dw
    return w0_new, w1_new



def run_gradient(x,y, alpha_max, max_iterations, stopping_threshold = 1e-16):
    dims = 1
    if len(x.shape) > 1:
        dims = x.shape[1]

    w1, w0 = initialise(dims)

    previous_cost = None
    cost_history = np.zeros(max_iterations)

    for itr in range(max_iterations):
        y_hat = predict(x, w1, w0)
        cost = compute_cost(x, y, y_hat)

        # early stopping criteria 
        if previous_cost and abs(previous_cost - cost) <= stopping_threshold:
            break

        cost_history[itr] = cost

        previous_cost = cost
        old_w1 = w1
        old_w0 = w0

        w1, w0 = update_parameters(x, y, y_hat, cost, w1, w0)


    return w1, w0, cost_history




def run_algo(x, y, alpha, num_iterations):
    w0, w1 = initialise(x.shape[1])
    num_iterations = 0
    gd_iterations_df = 10
    
    