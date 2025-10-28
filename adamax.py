import sys

import random

import numpy as np

import mnist_loader

import time

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    return ( 1 - sigmoid(x) ) * sigmoid(x)

class AdaMaxOpt(object):
    
    def __init__(self,sizes,eta,beta1=0.9,beta2=0.999,eps=1e-8):
        self.sizes = sizes
        
        self.eta = eta
        
        self.n_layers = len(sizes) - 2
        
        self.biases = [ np.random.randn(n, 1) for n in sizes[1:] ]
        
        self.weights = [ np.random.randn(a, b) for a, b in zip(sizes[:-1], sizes[1:]) ]
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = eps
        
        self.m_weights = [np.zeros_like(w) for w in self.weights]
        self.u_weights = [np.zeros_like(w) for w in self.weights]
        self.m_biases = [np.zeros_like(b) for b in self.biases]
        self.u_biases = [np.zeros_like(b) for b in self.biases]
        
        self.t = 0 
        
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

        for l in range(self.n_layers - 1, -1, -1):

            delta[l] = np.dot( self.weights[l + 1], delta[l + 1] ) \
                * sigmoid_prime(s[l + 1])

        return delta
    
    def step(self, activations, deltas):
        
        gradient_W = [ np.zeros(w.shape) for w in self.weights ]
        gradient_b = [ np.zeros(b.shape) for b in self.biases ]
        
        num_input = len(activations)

        for k in range(num_input):
            a = activations[k]
            d = deltas[k]
            
            for l in range(self.n_layers,-1,-1):
                gradient_W[l] += np.dot(a[l], d[l].T)
                gradient_b[l] += d[l]
                
        gradient_W = [gw / num_input for gw in gradient_W]
        gradient_b = [gb / num_input for gb in gradient_b] 
        
        self.t += 1
        
        for l in range(self.n_layers):
            self.m_weights[l] = self.beta1 * self.m_weights[l] + (1 - self.beta1) * gradient_W[l]
            self.u_weights[l] = np.maximum(self.beta2 * self.u_weights[l], np.abs(gradient_W[l]))
            m_hat_w = self.m_weights[l] / (1 - self.beta1 ** self.t)
            self.weights[l] -= (self.eta / (self.u_weights[l] + self.epsilon)) * m_hat_w
            
            self.m_biases[l] = self.beta1 * self.m_biases[l] + (1 - self.beta1) * gradient_b[l]
            self.u_biases[l] = np.maximum(self.beta2 * self.u_biases[l], np.abs(gradient_b[l]))
            m_hat_b = self.m_biases[l] / (1 - self.beta1 ** self.t)
            self.biases[l] -= (self.eta / (self.u_biases[l] + self.epsilon)) * m_hat_b
    
    
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
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                activations = []
                deltas = []
                for x, y in mini_batch:
                    a, s = self.forward(x)
                    delta = self.backward(a, s, y)
                    activations.append(a)
                    deltas.append(delta)
                self.step(activations, deltas)
                
            end = time.time()
            count = self.test(test_data)
            
            self.epoch_accuracy.append(count / len(test_data))
            self.epoch_times.append(end - start)
            print("Epoch", str(i), ":", str(count), "/", str(len(test_data)))


'''def main(ARGS):
    
    import matplotlib.pyplot as plt

    training_data, _, test_data = mnist_loader.load_data_wrapper()

    net = AdamOpt([784, 60, 30, 10], 0.002)

    net.train(training_data, test_data, 10, 30)
    
    plt.plot(range(1,11),net.epoch_times)
    plt.show()

    return 0


if __name__ == "__main__":

    ARGS = sys.argv[1:]

    RETCODE = main(ARGS)

    sys.exit(RETCODE)'''
    
        