#!/usr/bin/env python

# Run logistic regression training.
import random

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import assignment2 as a2

# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_epoch = 500
tol = 0.00001

# Step size for gradient descent.
eta_list = [0.5, 0.3, 0.1, 0.05, 0.01]

# Load data.
data = np.genfromtxt('data.txt')

# Data matrix, with column of ones at end.
X = data[:, 0:3]
# Target values, 0 for class 1, 1 for class 2.
t = data[:, 3]
# For plotting data
class1 = np.where(t == 0)
X1 = X[class1]
class2 = np.where(t == 1)
X2 = X[class2]



# Error values over all iterations.
e_all = [[],[],[],[],[]]

DATA_FIG = 1


# Set up the slope-intercept figure
# SI_FIG = 2
# plt.figure(SI_FIG)
plt.rcParams.update({'font.size': 15})
# plt.title('Separator in slope-intercept space')
# plt.xlabel('slope')
# plt.ylabel('intercept')
# plt.axis([-5, 5, -10, 0])
plt.figure()
for i in range(0, len(eta_list)) :
    # Initialize w.
    w = np.array([0.1, 0, 0])
    for epoch in range(0, max_epoch):
        sgd_order = np.arange(0, t.shape[0])
        random.shuffle(sgd_order)
        e=0.0
        for itr in sgd_order:
            # Compute output using current w on all data X.
            y_sgd = sps.expit(np.dot(X[itr], w))
            t_sgd = t[itr]
            #print('y_sgd & t_sgd', y_sgd,t_sgd)
            # e is the error, negative log-likelihood (Eqn 4.90)
            if(y_sgd==1 or y_sgd==0 ):
                temp = 0
            else:
                temp = -(np.multiply(t_sgd, np.log(y_sgd)) + np.multiply((1 - t_sgd), np.log(1 - y_sgd)))
            #print('temp',temp)
            e = e+temp
            #e = -np.mean(np.multiply(t, np.log(y)) + np.multiply((1 - t), np.log(1 - y)))
            # Gradient of the error, using Eqn 4.91
            grad_e = np.multiply((y_sgd - t_sgd), X[itr].T)
            #grad_e = np.mean(np.multiply((y - t), X.T), axis=1)
            # Update w, *subtracting* a step in the error derivative since we're minimizing
            w_old = w
            w = w - eta_list[i] * grad_e
        e = e/t.shape[0]+1
        e_all[i].append(e)
        if epoch > 0:
            if np.absolute(e - e_all[i][epoch - 1]) < tol:
                break
        print('epoch {0:d}, negative log-likelihood {1:.4f}, w={2}'.format(epoch, e, w.T))
    plt.plot(e_all[i])


plt.legend(['eta=0.5','eta=0.3','eta=0.1','eta=0.05','eta=0.01'])

plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression')
plt.xlabel('Epoch')
plt.show()
