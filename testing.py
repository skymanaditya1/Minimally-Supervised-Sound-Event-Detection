import os
import subprocess
import re
import numpy as np
from StringIO import StringIO
import pickle

WRITE_DIR ="/home/aditya/Desktop/CCBDNeuralNetworks/Training_Data/testing"
OPENSMILE_DIR = "/home/aditya/CCBD_Sound_Internship/openSMILE-2.1.0/bin/linux_x64_standalone_static"
TEST_DATASET = "/home/aditya/Desktop/CCBDNeuralNetworks/Training_Data"

# Calculates Number of instances in the file to test
def getNumInstances(filename):
	# Method to get the number of instances in a given arff file
	num = 0
	s = "java -cp /usr/share/java/weka.jar weka.core.Instances "+filename
	out = subprocess.check_output(s, shell=True)
	out_str = out.decode('utf-8')
	m = re.search(r'(Num Instances:)?(\d+).*', out_str)
	if (m):
		print m.group(0)
		num = int(m.group(2))
	else:
		print "Could not generate the number of instances"
		print "Number of instances : " + str(out_str)

	return num

# Generates the plaintext file
def generate_plaintext(file_to_read, file_to_write):
	f1 = open(file_to_read, 'r')
	f2 = open(file_to_write, 'w')

	line = f1.readline()
	while line != '@data'+'\n':
		line = f1.readline()
	f1.readline()

	lines = f1.readlines()
	for line in lines:
		f2.write(line)

	# Close the files
	f1.close()
	f2.close()

# Generates a file without commas
def create_commalessFile(filename, commalessFile):
	with open(filename, 'r') as data:
		plaintext = data.read()

	plaintext = plaintext.replace(',', ' ')
	os.system("touch " + commalessFile)
	ftw = open(commalessFile, 'w')
	for line in plaintext:
		ftw.write(line)

# Generates the numpy.ndarray for testing
def create_ndarray(filename):
	with open(filename, 'r') as data:
		plaintext = data.read()
	c = StringIO(plaintext)
	num_array = np.loadtxt(c, dtype='float')
	return num_array

def create_output(num_instances, nominal_val):
	num_arr = np.full(num_instances, nominal_val)
	return num_arr

# Method creates test_data in the mnist format
def load_new_data(test_data):
	test_inputs = [np.reshape(x, (39, 1)) for x in test_data[0]]
	new_test_data = zip(test_inputs, test_data[1])

	return new_test_data

def vectorized_result(j, num):
	# Helper method for load_data, to create a numpy array for the output
	e = np.zeros((num, 1))
	e[j] = 1.0
	return e

def load_testdata():
	print("Neural Network Testing")
	wav_file = input("Enter the path of the test wav file : ")
	# nom_values = display_nominal_data()
	display_nominal_data()
	nominal_val = int(input("Enter the output neuron value : "))
	string = OPENSMILE_DIR+"/SMILExtract -C " + OPENSMILE_DIR+"/MFCC12_E_D_A.conf -I " + wav_file + " -O " + WRITE_DIR + "/test.arff"
	os.system(string)

	# Returns the number of instances in the given file
	num_instances = getNumInstances(WRITE_DIR+"/test.arff")

	# A plain text file without the arff attributes
	generate_plaintext(WRITE_DIR+"/test.arff", WRITE_DIR+"/test")	

	# Creates a file without the commas
	create_commalessFile(WRITE_DIR+"/test", WRITE_DIR+"/test_plain")

	# Create numpy.ndarray from file_commaless
	ndarray_input = create_ndarray(WRITE_DIR+"/test_plain")

	# Generate the result of each instance
	ndarray_result = create_output(num_instances, nominal_val)

	# Testing data is prepared in a format similar to mnist
	testing_data = (ndarray_input, ndarray_result)

	# return testing_data
	new_testing_data = load_new_data(testing_data)

	# Pickle the testing_data for later use in Neural Networks
	pickle.dump(new_testing_data, open(TEST_DATASET+"/"+"testing_data.pkl", "wb"))

def display_nominal_data():
	file_path = TEST_DATASET+"/"+"nominal_sound_mapping.pkl"
	dic = pickle.load(open(file_path, 'rb'))

	for key in dic:
		print(key, dic[key])
	# return dic