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

class NeuralNetwork:
    def __init__(self):
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

        self.h1 = Neuron([self.w1, self.w2], self.b1) # output of h1
        self.h2 = Neuron([self.w3, self.w4], self.b2) # output of h2
        self.o1 = Neuron([self.w5, self.w6], self.b3) # output of o1 (final output)

    def feedforward(self, inputs):
        out_h1 = self.h1.feedforward(inputs)
        out_h2 = self.h2.feedforward(inputs)
        out_o1 = self.o1.feedforward([out_h1, out_h2])

        return out_o1

    def train(self, data, labels, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            for x, y in zip(data, labels):
                # Forward pass
                out_h1 = self.h1.feedforward(x)
                out_h2 = self.h2.feedforward(x)
                out_o1 = self.o1.feedforward([out_h1, out_h2])

                # Calculate error
                error = MSE(y, out_o1)

                # Backprop
                # w1h1
                w1_dLdy = -2 * (y - out_o1)
                w1_dydh = self.w5 * (out_o1 * (1 - out_o1))
                w1_dhdw = x[0] * (out_h1 * (1 - out_h1))
                w1_grad = w1_dLdy * w1_dydh * w1_dhdw
                self.w1 -= learning_rate * (w1_grad)
                # w2h1
                # b1h1

                # w3h2
                # w4h2
                # b2h2

                # w5o1
                # w6o1
                # b3o1

                pass

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





def MSE(y_true, y_pred): # y_true and y_pred are numpy arrays with the same length
    return ((y_true - y_pred) ** 2).mean()

data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork()