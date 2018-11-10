import numpy 
import scipy.special

class NeuralNetwork:
   
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        self.lr = learningrate

        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass

    
    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        return final_outputs

    
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

class_names = ['Iris-setosa\n', 'Iris-versicolor\n', 'Iris-virginica\n']

train_file_ = open("dataset/iris.csv", 'r')
train_data = numpy.array('')
train_data = train_file_.readlines()
train_file_.close()


input_nodes = 2
hidden_nodes = 3
output_nodes = 3

learning_rate = 0.12

n = NeuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

epochs = 10000

for i in range(epochs):
    for record in train_data:
        all_values = record.split(',')
        if '\n' not in all_values[:4]:
            inputs = (numpy.asfarray(all_values[2:4])/ 100.0*0.99) + 0.01
            targets = numpy.zeros(output_nodes) + 0.01
            targets[class_names.index(all_values[4])] = 0.99
            n.train(inputs, targets)
        pass


# Test with training data, change with test data or do by input

test_file = open("dataset/iris.csv", 'r')
test_data = numpy.array('')
test_data = test_file.readlines()
test_file.close()


test_input = test_data[0].split(',')
print('ZIEL:', test_data[0])
print('OUTPUT:', n.query(((numpy.asfarray(test_input[2:4]) / 100.0*0.99) + 0.01)))

test_input = test_data[66].split(',')
print('ZIEL:', test_data[66])
print('OUTPUT:', n.query(((numpy.asfarray(test_input[2:4]) / 100.0*0.99) + 0.01)))


test_input = test_data[130].split(',')
print('ZIEL:', test_data[130])
print('OUTPUT:', n.query(((numpy.asfarray(test_input[2:4]) / 100.0*0.99) + 0.01)))

