'''
network.py
Module implements Stochastic Gradient Descent Learning Algorithm.
Uses Feedforward neural network with backpropagation.
'''

import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        '''The list sizes contains the number of neurons in the 
        layers of the network. The first and last element specifies
        the number of neurons in the input layer and the output layer.
        All the intermediate numbers specifies the number of neurons
        in the intermediate hidden layers.
        '''
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        '''Returns a'''
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        '''Trains the neural network using mini-batch stochastic 
        gradient descent algorithm. The training data is a list
        of tuples in the mnist format. (x, y) represents training
        inputs and the desired outputs. 
        epochs is the number of iterations to traing the network.
        mini_batch_size is the size of each mini_batch
        eta is the learning rate of the gradient descent algorithm.
        test_data is the test_data
        '''
        
        # Accumulators for storing the average accuracy and accumulator
        accumulator = 0
        average_accuracy = 0
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
        average_accuracy = (accumulator * 100 ) / (epochs * n_test)
        print "The average accuracy over {0} epochs is : {1}".format(epochs, average_accuracy)

    def update_mini_batch(self, mini_batch, eta):
        '''Updates the network's weights and biases by applying
        the gradient descent algorithm using backpropagation to 
        a single mini batch of size mini_batch_size. 
        '''
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
        '''Returns a tuple representing the gradient descent cost function.
        '''
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

    def evaluate(self, test_data):
        '''Returns the number of test inputs for which the neural
        network outputs the correct result. The correct
        result corresponds to the output neuron which achieves the
        highest degree of activation.
        '''
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        
        # nom = load_nominal_sound_mappings()
        nom = {0:"cat", 1:"dog", 2:"whistle"}
        for (x, y) in test_data:
            argm = np.argmax(self.feedforward(x))
            print "Predicted Value : {0}, Sound : {1} Actual Value : {2}, Sound : {3}".format(argm, nom[argm], y, nom[int(y)])
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        # Returns the output activations
        return (output_activations-y)

# Sigmoidal function
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# Derivative of the sigmoidal function
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

# Returns the dictionary of nominal_value to sound_type mappings file
def load_nominal_sound_mappings():
    nominal_file = "/home/aditya/Desktop/CCBDNeuralNetworks/Training_Data/nominal_sound_mapping.pkl"
    dic = pickle.load(open(nominal_file), "rb")