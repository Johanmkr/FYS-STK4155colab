import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(-10, 10, num=1000)
x_data = x_data[:,np.newaxis]
y_data = 0.1*x_data*np.cos(x_data) + 0.1*np.random.normal(size=(1000,1))

y_data = (y_data-np.mean(y_data))/np.std(y_data)
x_data = (x_data-np.mean(x_data))/np.std(x_data)
plt.plot(x_data,y_data, ".")
plt.show()




def sigmoid(x):
    # if x<1e-5:
    #     return 0
    # elif x>1e5:
    #     return 1
    # else:
    return 1/(1 + np.exp(-x))


class NeuralNetwork:
    def __init__(
            self,
            X_data,
            Y_data,
            n_hidden_neurons=10,
            n_categories=10,
            epochs=5,
            batch_size=100,
            eta=0.1,
            lmbd=0.0):

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = len(x_data)

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def feed_forward(self):
        # feed-forward for training
        self.z_h = np.matmul(self.X_data, self.hidden_weights) + self.hidden_bias
        self.a_h = sigmoid(self.z_h)

        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias

        # exp_term = np.exp(self.z_o)
        # self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        self.output = sigmoid(self.z_o)

    def feed_forward_out(self, X):
        # feed-forward for output
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = sigmoid(z_h)

        z_o = np.matmul(a_h, self.output_weights) + self.output_bias
        
        # exp_term = np.exp(z_o)
        # probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        # return probabilities
        self.output = sigmoid(self.z_o) 
        return self.output

    def backpropagation(self):
        # print(self.z_o)
        error_output = self.z_o - self.Y_data
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.a_h * (1 - self.a_h)

        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient

    def predict(self, X):
        out = self.feed_forward_out(X)
        return out

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()

if __name__=="__main__":
    net = NeuralNetwork(x_data, y_data)
    net.train()

    xx = net.feed_forward_out(x_data)