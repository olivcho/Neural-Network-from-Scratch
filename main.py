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
weights = np.array([0, 1])
bias = 4
inputs = np.array([3, 4])

n = Neuron(weights, bias)
print(f"2-Input Neuron Output: {n.feedforward(inputs)}") # 0.999664649

class NeuralNetwork:
    def __init__(self):
        weights = np.array([0, 1]) # Assign all neurons the same weights
        bias = 0 # Assign all neurons the same bias

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, inputs):
        out_h1 = self.h1.feedforward(inputs)
        out_h2 = self.h2.feedforward(inputs)
        out_o1 = self.o1.feedforward([out_h1, out_h2])

        return out_o1
    
# Example usage for neural network with 1 hidden layer and 2 neurons
inputs = np.array([2, 3])

n = NeuralNetwork()
print(f"Neuron Network Output: {n.feedforward(inputs)}")

# MSE Loss Function
def MSE(y_true, y_pred): # y_true and y_pred are numpy arrays with the same length
    return ((y_true - y_pred) ** 2).mean()

# Example usage of MSE
print(MSE(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0])))

