
import numpy as np
inputs = np.array([1,2,3])
w_hidden1 = np.array([1,1,-5])
w_hidden2 = np.array([3,-4,2])
w_output = np.array([2,-1])

# First intermediary node is the sum of the products of the input times the
# weigths
n_hidden1 = sum(inputs * w_hidden1)
print 'n_hidden1: {}'.format(n_hidden1)
n_hidden2 = sum(inputs * w_hidden2)
print 'n_hidden1: {}'.format(n_hidden2)

# Those two results create 2 different nodes, so we create the new array
n_hidden_tot = np.array([n_hidden1, n_hidden2])
print 'n_hidden_tot: {}'.format(n_hidden_tot)

# Multiply the intermediary array times the w_outputs and add them together
# so we get the result.
result = sum(n_hidden_tot * w_output)
print 'result: {}'.format(result)
