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

        self.update_neurons()
    
    def update_neurons(self):
        self.h1 = Neuron([self.w1, self.w2], self.b1) # output of h1
        self.h2 = Neuron([self.w3, self.w4], self.b2) # output of h2
        self.o1 = Neuron([self.w5, self.w6], self.b3) # output of o1 (final output)

    def feedforward(self, inputs):
        out_h1 = self.h1.feedforward(inputs)
        out_h2 = self.h2.feedforward(inputs)
        out_o1 = self.o1.feedforward([out_h1, out_h2])

        return out_o1

    def train(self, data, labels, epochs=10000, learning_rate=0.01):
        for epoch in range(epochs):
            loss = 0
            
            for x, y in zip(data, labels):
                # Forward pass
                out_h1 = self.h1.feedforward(x)
                out_h2 = self.h2.feedforward(x)
                out_o1 = self.o1.feedforward([out_h1, out_h2])

                # Calculate error for this sample
                error = MSE(y, out_o1)
                loss += error

                # Backprop
                # w1h1
                w1_dLdy = -2 * (y - out_o1)
                w1_dydh = self.w5 * (out_o1 * (1 - out_o1))
                w1_dhdw = x[0] * (out_h1 * (1 - out_h1))
                w1_grad = w1_dLdy * w1_dydh * w1_dhdw
                # w2h1
                w2_dLdy = -2 * (y - out_o1)
                w2_dydh = self.w5 * (out_o1 * (1 - out_o1)) 
                w2_dhdw = x[1] * (out_h1 * (1 - out_h1))
                w2_grad = w2_dLdy * w2_dydh * w2_dhdw
                # b1h1
                b1_dLdy = -2 * (y - out_o1)
                b1_dydh = self.w5 * (out_o1 * (1 - out_o1)) 
                b1_dhdb = (out_h1 * (1 - out_h1))
                b1_grad = b1_dLdy * b1_dydh * b1_dhdb

                # w3h2
                w3_dLdy = -2 * (y - out_o1)
                w3_dydh = self.w6 * (out_o1 * (1 - out_o1))
                w3_dhdw = x[0] * (out_h2 * (1 - out_h2))
                w3_grad = w3_dLdy * w3_dydh * w3_dhdw

                # w4h2
                w4_dLdy = -2 * (y - out_o1)
                w4_dydh = self.w6 * (out_o1 * (1 - out_o1))
                w4_dhdw = x[1] * (out_h2 * (1 - out_h2))
                w4_grad = w4_dLdy * w4_dydh * w4_dhdw

                # b2h2
                b2_dLdy = -2 * (y - out_o1)
                b2_dydh = self.w6 * (out_o1 * (1 - out_o1))
                b2_dhdb = (out_h2 * (1 - out_h2))
                b2_grad = b2_dLdy * b2_dydh * b2_dhdb

                # w5o1
                w5_dLdy = -2 * (y - out_o1)
                w5_dydo = out_o1 * (1 - out_o1)
                w5_dodw = out_h1
                w5_grad = w5_dLdy * w5_dydo * w5_dodw

                # w6o1
                w6_dLdy = -2 * (y - out_o1)
                w6_dydo = out_o1 * (1 - out_o1)
                w6_dodw = out_h2
                w6_grad = w6_dLdy * w6_dydo * w6_dodw

                # b3o1
                b3_dLdy = -2 * (y - out_o1)
                b3_dydo = out_o1 * (1 - out_o1)
                b3_dodb = 1
                b3_grad = b3_dLdy * b3_dydo * b3_dodb

                self.w1 -= learning_rate * w1_grad
                self.w2 -= learning_rate * w2_grad
                self.b1 -= learning_rate * b1_grad
                self.w3 -= learning_rate * w3_grad
                self.w4 -= learning_rate * w4_grad
                self.b2 -= learning_rate * b2_grad
                self.w5 -= learning_rate * w5_grad
                self.w6 -= learning_rate * w6_grad
                self.b3 -= learning_rate * b3_grad

                self.update_neurons()

            if epoch % 10 == 0:
                print("Epoch %d loss: %.3f" % (epoch, loss))

def MSE(y_true, y_pred): # y_true and y_pred are numpy arrays with the same length
    return ((y_true - y_pred) ** 2).mean()

# Create data points array
data = np.array([
    [4.2, -3.1],
    [-2.5, 5.8],
    [6.7, 7.2],
    [-8.1, -2.3],
    [0.5, 1.8],
    [-5.4, 4.2],
    [3.1, -7.5],
    [-1.2, -4.6],
    [7.8, 2.5],
    [-4.3, -6.2],
    [2.9, 8.1],
    [-7.6, 3.4],
    [1.7, -5.9],
    [-3.8, -1.6],
    [5.2, 6.3],
    [-6.7, -8.4],
    [8.5, -2.7],
    [-9.3, 7.1],
    [0.3, -8.9],
    [9.6, 9.2]
])

# Create corresponding labels
# Pattern: label=1 if (x^2 + y^2 < 36) or (x>2 and y>2), else label=0
labels = np.array([
    0,  # [4.2, -3.1]
    0,  # [-2.5, 5.8]
    1,  # [6.7, 7.2]
    0,  # [-8.1, -2.3]
    1,  # [0.5, 1.8]
    0,  # [-5.4, 4.2]
    0,  # [3.1, -7.5]
    0,  # [-1.2, -4.6]
    1,  # [7.8, 2.5]
    0,  # [-4.3, -6.2]
    1,  # [2.9, 8.1]
    0,  # [-7.6, 3.4]
    0,  # [1.7, -5.9]
    1,  # [-3.8, -1.6]
    1,  # [5.2, 6.3]
    0,  # [-6.7, -8.4]
    0,  # [8.5, -2.7]
    0,  # [-9.3, 7.1]
    0,  # [0.3, -8.9]
    1   # [9.6, 9.2]
])

nn = NeuralNetwork()
nn.train(data, labels)

print(nn.feedforward(np.array([9.6, 9.2])))