# Import the necessary libraries
import numpy
import scipy.special

# Define a NeuralNetwork class
class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # Initialize the neural network with input nodes, hidden nodes, output nodes, and learning rate
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # Initialize the weight matrices with random values
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        
        # Set the learning rate
        self.lr = learningrate
        
        # Define the activation function (sigmoid function)
        self.activation_function = lambda x: scipy.special.expit(x)

    # Train the neural network
    def train(self, inputs_list, targets_list):
        # Convert input and target lists to NumPy arrays and transpose them
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # Calculate hidden layer inputs and outputs
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Calculate final layer inputs and outputs
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        # Calculate output errors and hidden errors
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        
        # Update the weights using gradient descent
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        # Return the final outputs
        return final_outputs

    # Query the neural network
    def query(self, inputs_list):
        # Convert input list to a NumPy array and transpose it
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # Calculate hidden layer inputs and outputs
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Calculate final layer inputs and outputs
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        # Return the final outputs
        return final_outputs

# Define class names for classification
class_names = ['Iris-setosa\n', 'Iris-versicolor\n', 'Iris-virginica\n']

# Read training data from a file
train_file_ = open("dataset/iris.csv", 'r')
train_data = numpy.array('')
train_data = train_file_.readlines()
train_file_.close()

# Set the neural network parameters
input_nodes = 2
hidden_nodes = 3
output_nodes = 3
learning_rate = 0.12

# Initialize the neural network
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Training loop
epochs = 10000
for i in range(epochs):
    for record in train_data:
        all_values = record.split(',')
        if '\n' not in all_values[:4]:
            # Preprocess input and target data and train the network
            inputs = (numpy.asfarray(all_values[2:4]) / 100.0 * 0.99) + 0.01
            targets = numpy.zeros(output_nodes) + 0.01
            targets[class_names.index(all_values[4])] = 0.99
            n.train(inputs, targets)

# Test the trained neural network
test_file = open("dataset/iris.csv", 'r')
test_data = numpy.array('')
test_data = test_file.readlines()
test_file.close()

# Test with specific data points
test_input = test_data[0].split(',')
print('ZIEL:', test_data[0])
print('OUTPUT:', n.query(((numpy.asfarray(test_input[2:4]) / 100.0 * 0.99) + 0.01)))

print('ZIEL:', test_data[0])
print('OUTPUT:', n.query(((numpy.asfarray(test_input[2:4]) / 100.0*0.99) + 0.01)))

test_input = test_data[66].split(',')
print('ZIEL:', test_data[66])
print('OUTPUT:', n.query(((numpy.asfarray(test_input[2:4]) / 100.0*0.99) + 0.01)))

test_input = test_data[130].split(',')
print('ZIEL:', test_data[130])
print('OUTPUT:', n.query(((numpy.asfarray(test_input[2:4]) / 100.0*0.99) + 0.01)))
