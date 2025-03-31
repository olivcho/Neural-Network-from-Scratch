import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias  # Calculate the weighted sum
        return sigmoid(total)

# Example usage for 2-input neuron
weights = [0, 1]
bias = 4
inputs = np.array([3, 4])

n = Neuron(weights, bias)
print(n.feedforward(inputs)) # 0.999664649

