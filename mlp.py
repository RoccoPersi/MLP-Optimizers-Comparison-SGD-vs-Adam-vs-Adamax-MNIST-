import sys

import random

import numpy as np

import mnist_loader

import time

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    return ( 1 - sigmoid(x) ) * sigmoid(x)

class Network(object):

    def __init__(self, sizes, eta):
        
        self.sizes = sizes
        
        self.eta = eta

        # numero di layer
        self.num_layers = len(sizes) - 2

        # bias
        self.biases = [ np.random.randn(n, 1) for n in sizes[1:] ]
        # self.biases = []
        # for n in sizes[1:]:
        #     self.biases.append( np.random.randn(n, 1) )

        # weights
        self.weights = [ np.random.randn(a, b) for a, b in zip(sizes[:-1], sizes[1:]) ]
        
        self.epoch_times = []
        
        self.epoch_accuracy = []
        
    def get_EpochTimes(self):
        return self.epoch_times
    
    def get_EpochAccuracy(self):
        return self.epoch_accuracy

    def forward(self, x):

        a = [x]
        s = [0]

        for W, b in zip(self.weights, self.biases):

            s.append(np.dot( np.transpose(W), a[-1] ) + b)

            a.append( sigmoid(s[-1]) )
        
        # return a[-1]
        return a, s

    def backward(self, a, s, y):
        
        delta = [ np.zeros((n, 1)) for n in self.sizes[1:] ]

        delta[-1] = (a[-1] - y) * sigmoid_prime(s[-1])

        for l in range(self.num_layers - 1, -1, -1):

            delta[l] = np.dot( self.weights[l + 1], delta[l + 1] ) \
                * sigmoid_prime(s[l + 1])

        return delta

    def step(self, activations, deltas):
        
        gradient_W = [ np.zeros(w.shape) for w in self.weights ]
        gradient_b = [ np.zeros(b.shape) for b in self.biases ]

        num_input = len(activations)

        for l in range(self.num_layers, -1, -1):

            sum_w = 0
            for k in range(num_input):
                sum_w += np.dot( activations[k][l], np.transpose( deltas[k][l] ) )

            sum_b = 0
            for k in range(num_input):
                sum_b += deltas[k][l]

            gradient_W[l] = sum_w
            gradient_b[l] = sum_b

        self.weights = [
            w - (self.eta / num_input) * dw
            for w, dw in zip( self.weights, gradient_W )
        ]

        self.biases = [
            b - (self.eta / num_input) * db
            for b, db in zip( self.biases, gradient_b )
        ]


    def test(self, test_data):
        
        count = 0

        for x, y in test_data:
            
            a, _ = self.forward(x)

            output = a[-1]

            if np.argmax(output) == y:
                count += 1

        return count


    def train(self, training_data, test_data, n_epochs, mini_batch_size):
            
        n = len(training_data)

        for i in range(n_epochs):
            start = time.time()

            random.shuffle(training_data)

            mini_batches = [ training_data[k : k + mini_batch_size] 
                            for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:

                activations = []
                deltas = []

                for x, y in mini_batch:
                # for x, y in training_data:

                    a, s = self.forward(x)

                    delta = self.backward(a, s, y)

                    activations.append(a)
                    deltas.append(delta)

                self.step(activations, deltas)
                
                end = time.time()

            count = self.test(test_data)
            self.epoch_accuracy.append(count/len(test_data))
            
            self.epoch_times.append(end-start)
            print("Epoch", str(i), ":", str(count), "/", str(len(test_data)))


'''def main(ARGS):

    training_data, _, test_data = mnist_loader.load_data_wrapper()

    net = Network([784, 60, 30, 10], 0.1)

    net.train(training_data, test_data, 10, 30)

    return 0


if __name__ == "__main__":

    ARGS = sys.argv[1:]

    RETCODE = main(ARGS)

    sys.exit(RETCODE)'''