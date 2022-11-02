import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(-10, 10, num=1000)
x_data = x_data[:,np.newaxis]
y_data = 0.1*x_data*np.cos(x_data) + 0.1*np.random.normal(size=(1000,1))

y_data = (y_data-np.mean(y_data))/np.std(y_data)
x_data = (x_data-np.mean(x_data))/np.std(x_data)
plt.plot(x_data,y_data, ".", color="blue",label="data")
# plt.show()
# print(x_data.T.shape)
# print(y_data.T.shape)



#franke data
# Make data.
# space = np.linspace(0,1,20)
# xx, yy = np.meshgrid(space,space)

# def FrankeFunction(x,y):
#     term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
#     term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
#     term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
#     term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
#     return term1 + term2 + term3 + term4

# zz = FrankeFunction(xx, yy)
# zzr = zz.ravel()
# zzr = (zzr-np.mean(zzr))/np.std(zzr)
# FrankeX = np.zeros((2,len(zzr)))
# FrankeX[0,:] = xx.ravel()
# FrankeX[1,:] = yy.ravel()
# FrankeY = zzr[:,np.newaxis].T





def sigmoid(x):
    # if x<1e-5:
    #     return 0
    # elif x>1e5:
    #     return 1
    # else:
    return 1/(1 + np.exp(-x))

def sigmoid_deriv(z):
    return (sigmoid(z) * (1-sigmoid(z)))

class NeuralNetwork:
    def __init__(
            self,
            X_data,
            Y_data,
            n_hidden_neurons=10,
            output_length=10,
            epochs=5,
            batch_size=100,
            eta=0.1,
            lmbd=0.0):

        self.X_data_full = X_data
        self.Y_data_full = Y_data
        
        self.X_data = X_data    #for testing
        self.Y_data = Y_data    #for testing

        # self.n_inputs = X_data.shape[0]
        self.n_features = len(X_data)
        self.n_hidden_neurons = n_hidden_neurons
        self.output_length = output_length

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.output_length)
        self.output_bias = np.zeros(self.output_length) + 0.01

    def feed_forward_train(self):
        # feed-forward for training
        self.z_h = self.X_data @ self.hidden_weights + self.hidden_bias
        self.a_h = sigmoid(self.z_h)
        self.z_o = self.a_h @ self.output_weights + self.output_bias
        self.output = (self.z_o-np.mean(self.z_o))/np.std(self.z_o)

    def feed_forward_out(self, X):
        # feed-forward for output
        z_h = X @ self.hidden_weights + self.hidden_bias
        a_h = sigmoid(z_h)
        z_o = a_h @ self.output_weights + self.output_bias
        self.output = (z_o-np.mean(z_o))/np.std(z_o)
        return self.output

    def back_propagation(self):
        # print(self.z_o)
        error_output = self.z_o - self.Y_data
        error_hidden = error_output @ self.output_weights.T * sigmoid_deriv(self.a_h)

        self.output_weights_gradient = self.a_h.T @ error_output
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = self.X_data.T @ error_hidden
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

    # def train(self):
    #     data_indices = np.arange(self.n_inputs)

    #     for i in range(self.epochs):
    #         for j in range(self.iterations):
    #             # pick datapoints with replacement
    #             chosen_datapoints = np.random.choice(
    #                 data_indices, size=self.batch_size, replace=False
    #             )

    #             # minibatch training data
    #             self.X_data = self.X_data_full[chosen_datapoints]
    #             self.Y_data = self.Y_data_full[chosen_datapoints]

    #             self.feed_forward_train()
    #             self.back_propagation()


    def train(self, N):
        for _ in range(N):
            self.feed_forward_train()
            self.back_propagation()

if __name__=="__main__":
    n_hidden_neurons = 100
    batch_size = 100
    epochs = 100
    eta = 0.01
    lmbd = 0.1
    N = 1000
    net = NeuralNetwork(x_data.T, y_data.T, n_hidden_neurons=10, output_length=y_data.T.shape[1], epochs=epochs, batch_size=batch_size, eta=eta, lmbd=lmbd)
    net.train(N)
    # from IPython import embed; embed()
    prediction = net.predict(x_data.T).T
    # print(f"Iterations: {net.iterations}")
    plt.plot(x_data, prediction, color="red", label="prediction", alpha=0.7)
    plt.legend()
    plt.show()


    #Franke
    # FrankeNet = NeuralNetwork(FrankeX, FrankeY, n_hidden_neurons=10, output_length=FrankeY.shape[1], epochs=epochs, batch_size=batch_size, eta=eta, lmbd=lmbd)
    # FrankeNet.train(N)
    # FrankePred = FrankeNet.predict(FrankeX)
    # fig = plt.figure(figsize=(15,15))
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(xx, yy, np.reshape(FrankePred[0], (20,20)), cmap="coolwarm", alpha=0.7)
    # scatter = ax.scatter(xx,yy,zzr, color="green")
    # plt.show()
    # print(net.predict(np.array([[7.5]])))