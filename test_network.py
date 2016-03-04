'''
network.py
Module implements Stochastic Gradient Descent Learning Algorithm.
Uses Feedforward neural network with backpropagation.
'''

import random
import numpy as np
import pickle

WRITE_DIR="/home/aditya/Desktop/CCBDNeuralNetworks/Training_Data"

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = pickle.load(open(WRITE_DIR+"/trained_biases.pkl", "rb"))
        self.weights = pickle.load(open(WRITE_DIR+"/trained_weights.pkl", "rb"))

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self):
        test_path = input("Please provide the pkl file for testing : ")
        test_data = pickle.load(open(test_path, 'rb'))
        n_test = len(test_data)
        eval1 = self.evaluate(test_data)
        percentage_accuracy = eval1 * 100 / n_test
        print "Percentage Accuracy : {0}".format(percentage_accuracy)
    
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        
        # nom = load_nominal_sound_mappings()
        nom = {0:"air_conditioner", 1:"dog", 2:"drilling", 3:"gun", 4:"jackhammer", 5:"siren"}
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
    dic = pickle.load(open(nominal_file, "rb"))
    return dic
