import pickle
import gzip

import numpy as np

def load_data():
    """
    training_data is a tuple containing:
        - np.array with 50000 entries, each of them a np.array of 784 = 28 * 28 pixels
        - np.array with 50000 entries, each of them a digit in range 0..9
    
    test_data and validation_data are the same but with 10000 entries each
    """

    f = gzip.open('mnist.pkl.gz', 'rb')

    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")

    f.close()

    return (training_data, validation_data, test_data)


def load_data_wrapper():
    """
    training_data is a list of 50000 pairs (x, y) where
        - x is a np.array with 784 values
        - y is a one-hot label
    
    test_data and validation_data are lists of 10000 pairs (x, y) where
        - x is a np.array with 784 values
        - y is a digit in range 0..9
    """

    tr_d, va_d, te_d = load_data()

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]

    training_results = [vectorized_result(y) for y in tr_d[1]]

    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]

    validation_data = list(zip(validation_inputs, va_d[1]))

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]

    test_data = list(zip(test_inputs, te_d[1]))

    return (training_data, validation_data, test_data)


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere. This is used to convert a digit
    in 0..9 into the corresponding one-hot label."""

    e = np.zeros((10, 1))

    e[j] = 1.0

    return e
