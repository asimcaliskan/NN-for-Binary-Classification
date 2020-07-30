import numpy as np
import dataset_loader
import random

class Network:
    def __init__(self, network_list):
        """Number of Layers = Length of Network List*
          *Network List = [x,y,z...]"""
        self.num_layers = len(network_list)
        self.weight = [np.random.randn(k, j) for j, k in zip(network_list[:-1],network_list[1:])]#Gaussian distribution 
        self.bias   = [np.random.randn(i,1) for i in network_list[1:]]
    
    def Stochastic_Gradient_Descent(self, file_name, n_training_data, mini_batch_size, epoch_number, learning_rate):
        (training_data, test_data) = dataset_loader.separated_dataset(file_name, n_training_data)
        for i in range(epoch_number):
            random.shuffle(training_data)
            mini_batches = [training_data[i: i + mini_batch_size] for i in range(0, n_training_data, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_neural_network(mini_batch, learning_rate)
            print("Epoch {}: {} ".format(i, self.evaluate(test_data)/ len(test_data)))
            
            
    def backpropagation(self, x, y):
        derivative_w = [np.zeros(w.shape) for w in self.weight]
        derivative_b = [np.zeros(b.shape) for b in self.bias]
        activation = x
        activations= [x] 
        z_values = []
        for b, w in zip(self.bias, self.weight):
            z = np.dot(w, activation) + b
            z_values.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        error = self.derivative_quadratic_cost(activations[-1], y) * self.derivative_sigmoid(z_values[-1])
        derivative_b[-1] = error
        derivative_w[-1] = np.dot(error, activations[-2].transpose())
        
        for l in range(2, self.num_layers):
            error = np.dot(self.weight[-l + 1].transpose(), error) * self.derivative_sigmoid(z_values[-l])
            derivative_b[-l] = error
            derivative_w[-l] = np.dot(error, activations[-l - 1].transpose())
        return (derivative_b, derivative_w)
    
    def update_neural_network(self, mini_batch, learning_rate):
        derivative_w = [np.zeros(w.shape) for w in self.weight]
        derivative_b = [np.zeros(b.shape) for b in self.bias]

        for x, y in mini_batch:
            delta_derivative_b, delta_derivative_w = self.backpropagation(x, y)
            derivative_b = [db + ddb for db, ddb in zip(derivative_b, delta_derivative_b)]
            derivative_w = [dw + ddw for dw, ddw in zip(derivative_w, delta_derivative_w)]
        self.weight = [w - (learning_rate / len(mini_batch)) * dw for w, dw in zip(self.weight, derivative_w)]
        self.bias   = [b - (learning_rate / len(mini_batch)) * db for b, db in zip(self.bias , derivative_b)]
        
    def derivative_quadratic_cost(self, a, y):
        return a - y
    
    def feedforward(self, a):
        for w, b in zip(self.weight, self.bias): 
            a = self.sigmoid(np.dot(w, a) + b)
        return a
    
    def evaluate(self, test_data):
        test_results = [ (np.round(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def quadratic_cost(self, n, y):
        return np.sum()/2*n
    
    def sigmoid(self, x):
        return 1.0 /(1.0 + np.exp(-x))
    
    def derivative_sigmoid(self,x):
        return self.sigmoid(x)*(1 - self.sigmoid(x))
    
        