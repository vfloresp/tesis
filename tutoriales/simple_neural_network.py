from os import error
import numpy as np
from random import random

#save activations and derivatives
#implement back propagation
#implement gradient descent
#implement train 
#train our net with some dummy dataset
#make some predictions

class MLP(object):

    def __init__(self, num_imputs=3, num_hidden=[3,5], num_outputs=2):
        self.num_inputs = num_imputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        #initiate random weights
        weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i],layers[i+1])
            weights.append(w)
        self.weights = weights

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        derivatives = []
        for i in range(len(layers)-1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives


    def forward_propagate(self, inputs):
        activations = inputs
        self.activations[0] = inputs

        for i,w in enumerate(self.weights):
            #calculate net inputs
            net_inputs = np.dot(activations, w)

            #calculate the activations
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations

        return activations

    
    def back_propagate(self, error, verbose=False):
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivative(activations) # ndarray([0.1,0.2]) -->ndarray([[0.1,0.2]])
            delta_reshaped = delta.reshape(delta.shape[0],-1).T
            current_activations = self.activations[i] # ndarray([0.1,0.2]) -->ndarray([[0.1],[0.2]])
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0],-1)
            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)

            if(verbose):
                print("derivatives for W{}: {}".format(i,self.derivatives[i]))

        return error

    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            #print("Original W{} {}".format(i, weights))
            derivatives = self.derivatives[i]
            weights += derivatives*learning_rate
            #print("Updated W{} {}".format(i, weights))

    def train(self, inputs, targets, epochs, learning_rate):
        for i in range(epochs):
            sum_error = 0
            for (input,target) in zip(inputs,targets):
                #Create forward prop
                output = mlp.forward_propagate(input)

                #calculate error
                error = target - output

                #back propagation
                mlp.back_propagate(error)

                #Gradient descent
                mlp.gradient_descent(learning_rate=1)

                sum_error += self._mse(target, output)
            #report error
            print("error: {} at epoch {}".format(sum_error/len(inputs), i))


    def _mse(self, target, output):
        return np.average((target-output)**2)


    def _sigmoid_derivative(self,x):
        return x * (1.0 - x)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    # create an MLP model
    mlp = MLP(2,[5],1)

    #Create some inputs 
    input = np.array([0.1,0.2])
    target = np.array([0.3])

    inputs = np.array([[random()/2 for _ in range(2)] for _ in range(1000)]) # array([[0.1,0.2],[0.3,0.4]])
    targets = np.array([[i[0]+i[1]] for i in inputs])

    #train our mlp
    mlp.train(inputs, targets, 50, 0.1)
    
    #create dummy data
    input = np.array([0.3,0.1])
    target = np.array([0.4])

    output = mlp.forward_propagate(input)

    #print results
    print("The network input is {}".format(input))
    print("The network output is {}".format(output))
    print("the network is off by {}".format(target-output))