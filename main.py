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
        weights = np.array([0.5, 0.5]) # Assign all neurons the same weights
        bias = 0 # Assign all neurons the same bias

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, inputs):
        out_h1 = self.h1.feedforward(inputs)
        out_h2 = self.h2.feedforward(inputs)
        out_o1 = self.o1.feedforward([out_h1, out_h2])

        return out_h1, out_h2, out_o1 # out_h1 = out_h2 when weights and biases are the same

# Example usage for neural network with 1 hidden layer and 2 neurons
inputs = np.array([2, 3])

n = NeuralNetwork()
o1 = np.float64(n.feedforward(inputs))
print(f"Neuron Network Output: {o1[2]}")

# MSE loss function
def MSE(y_true, y_pred): # y_true and y_pred are numpy arrays with the same length
    return ((y_true - y_pred) ** 2).mean()

# Example usage of MSE
print(MSE(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0])))

# Backprop -- change in loss function with respect to weight 1
# dL/dw_1 = dL/dy * dy/dh_1 * dh_1/dw_1
# Recall that y = (1-y_pred)^2 by the MSE formula for one point
# Sigmoid: f(x) = 1/(1+exp(-x))
# Derivative: f'(x) = f(x) * (1-f(x))

# Example usage of backprop for previous NN example

dldy = -2 * (1 - o1[2]) # chain rule
dydh1 = 0.5 * (sigmoid(o1[2]) * (1 - sigmoid(o1[2]))) # chain rule on sigmoid function for o1
dh1dw1 = inputs[0] * (sigmoid(o1[0]) * (1 - sigmoid(o1[0]))) # chain rule on sigmoid function for h1

dldw = dldy * dydh1 * dh1dw1
print(f"Backprop: {dldw}")

# Training - Stochastic Gradient Descent
