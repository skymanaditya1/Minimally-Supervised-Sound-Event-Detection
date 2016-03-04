'''
network.py
Module implements Stochastic Gradient Descent Learning Algorithm.
Uses Feedforward neural network with backpropagation.
'''

import random
import numpy as np
import pickle

WRITE_DIR = "/home/aditya/Desktop/CCBDNeuralNetworks/Training_Data"

class Network(object):

    def __init__(self, sizes):

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, epochs, mini_batch_size, eta, test_data=None):

        # Accumulators for storing the average accuracy and accumulator
        accumulator = 0
        average_accuracy = 0
        training_path = input("Enter the path of the training data pickle file : ")
        training_data = pickle.load(open(training_path, 'rb'))
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                eval1 = self.evaluate(test_data)
                percentage_accuracy = eval1 * 100 / n_test
                print "Percentage Accuracy : {0}".format(percentage_accuracy)
                print "Epoch {0}: {1} / {2}".format(
                    j, eval1, n_test)
                print "Mini batch size : {0}".format(mini_batch_size)
                accumulator += eval1 
            else:
                print "Epoch {0} complete".format(j)
                print "Mini batch size : {0}".format(mini_batch_size)
        '''average_accuracy = (accumulator * 100 ) / (epochs * n_test)
        print "The average accuracy over {0} epochs is : {1}".format(epochs, average_accuracy)'''

        # Pickle the updated weights and biases, to be used for the test_data iteration
        pickle.dump(self.weights, open(WRITE_DIR+"/trained_weights.pkl", "wb"))
        pickle.dump(self.biases, open(WRITE_DIR+"/trained_biases.pkl", "wb"))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]    # list to store all the activations, layer by layer
        zs = []              # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        # Returns the output activations
        return (output_activations-y)

# Sigmoidal function
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# Derivative of the sigmoidal function
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))