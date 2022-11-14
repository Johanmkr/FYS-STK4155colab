from src.utils import *
from src.devNeuralNetwork import devNeuralNetwork as NeuralNet
from sklearn.datasets import load_breast_cancer


np.random.seed(69)



import src.infoFile_ala_Nanna as info
output_path = "../output/data/network_classification/"
info.init(path=output_path)
info.sayhello()

info.define_categories({
    "method":"method", 
    "opt":"optimiser", 
    "n_obs":r"$n_\mathrm{obs}$", 
    "no_epochs":"#epochs", 
    "no_minibatches":r"$m$",
    "eta":r"$\eta$", 
    "lmbda":r"$\lambda$",
    "L":r"$L-1$", 
    "N":r"$N_l$",
    "g":r"$g$",
    "timer":"train time (s)",
    "gamma":r"$\gamma$", 
    "rho":r"$\varrho_1$, $\varrho_2$",
    "theta0":r"$\theta_0$"
    })



class CancerData:
    def __init__(self,
                hiddenLayers = 0,
                neuronsInEachLayer = 1,
                activationFunction = 'sigmoid',
                outputFunction = 'sigmoid',
                outputNeurons = None,
                lossFunction = 'crossentropy',
                optimizer = 'RMSProp',
                epochs = None,
                batchSize = None,
                nrMinibatches = None,
                eta = 0.01,
                lmbda = 0.0,
                testSize = 0.2,
                terminalUpdate = False):

        cancer = load_breast_cancer()
        cancerpd = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        inputs = cancer.data 
        outputs = cancer.target 
        labels = cancer.feature_names[0:30]
        correlation_matrix = cancerpd.corr().round(1)

        self.inputs = inputs
        self.n_obs = len(outputs)

        self.Net = NeuralNet(inputs, outputs[:,np.newaxis], hiddenLayers=hiddenLayers, neuronsInEachLayer=neuronsInEachLayer, activationFunction=activationFunction, outputFunction=outputFunction, outputNeurons=outputNeurons, lossFunction=lossFunction,
        optimizer=optimizer, epochs=epochs, batchSize=batchSize, nrMinibatches=nrMinibatches, eta=eta, lmbda=lmbda, testSize=testSize,
        terminalUpdate = terminalUpdate, classification=True)

    def __str__(self):
        self.finalTestLoss()
        self.finalTrainLoss()
        print(self.Net)

        stringToReturn = f"Train accuracy:  {self.trainLoss:.2f}\n"
        stringToReturn += f"Test accuracy:  {self.testLoss:.2f}\n"
        return stringToReturn

    def train(self):
        # print(self.Net)
        t0 = time()
        self.Net.train()
        t1 = time()
        print(f"Training time: {t1-t0:.2f} s")

    def finalTestLoss(self):
        self.testLoss = self.Net.get_testLoss()
        return self.testLoss

    def finalTrainLoss(self):
        self.trainLoss = self.Net.get_trainLoss()
        return self.trainLoss

    def predict(self, X=None):
        data = X or self.inputs
        self.prediction = self.Net(data)
        return self.prediction

def EtaLambdaAnalysis(filename):
    #   Fixed parameters
    hiddenLayers = 1
    neuronsInEachLayer=5
    outputNeurons=1
    activationFunction = 'sigmoid'
    epochs=250
    nrMinibatches=5
    testSize=0.2
    optimizer='RMSProp'
    rho = (0.9,0.999) # default hyperparams in RMSProp

    #   Parameters to test
    etas = np.logspace(-8,1,10)
    lmbdas = np.logspace(-8,1,10)

    #   Paramteres to find
    accuracy = np.zeros((len(lmbdas), len(etas)))

    #   Testing loop
    for i, lmbda in enumerate(lmbdas):
        for j, eta in enumerate(etas):
            print(f"Lmbda: {lmbda}\nEta: {eta}")
            Creg = CancerData(hiddenLayers=hiddenLayers, neuronsInEachLayer=neuronsInEachLayer, activationFunction=activationFunction, outputNeurons=outputNeurons, optimizer=optimizer, epochs=epochs, nrMinibatches=nrMinibatches, eta=eta, lmbda=lmbda, testSize=testSize, terminalUpdate=False)
            Creg.train()
            print(Creg)
            print("\n\n\n\n")
            accuracy[i,j] = Creg.finalTestLoss()
    idx = [r"$10^{%.0f}$"%k for k in np.log10(lmbdas)]
    cols = [r"$10^{%.0f}$"%k for k in np.log10(etas)]
    dF = pd.DataFrame(accuracy, index=idx, columns=cols)
    dF.to_pickle(output_path+filename+".pkl")

    info.set_file_info(filename+".pkl", method='SGD', opt=optimizer, no_epochs=epochs, no_minibatches=nrMinibatches, L=hiddenLayers, eta=r"$[%s, %s]$" %(cols[0], cols[-1]), N=neuronsInEachLayer, lmbda=r"$[%s, %s]$" %(idx[0], idx[-1]), g=activationFunction, n_obs=Creg.n_obs, rho=rho)

def LayerNeuronsAnalysis(filename):
    #   Fixed parameters
    eta = 1e-3  #Based on previous results
    lmbda = 1e-4    #Based on previous results
    outputNeurons=1
    activationFunction = 'sigmoid'
    epochs=250
    nrMinibatches=5
    testSize=0.2
    optimizer='RMSProp'
    rho = (0.9,0.999) # default hyperparams in RMSProp

    #   Parameters to test
    layers = np.arange(10)
    neurons = np.arange(1,11)*5

    #   Parameters to find
    accuracy = np.zeros((len(layers), len(neurons)))

    #   Test loop
    for i, layer in enumerate(layers):
        for j, neuron in enumerate(neurons):
            print(f"Layer: {layer}\nNeuron: {neuron}")
            Creg = CancerData(hiddenLayers=layer, neuronsInEachLayer=neuron, activationFunction=activationFunction, outputNeurons=outputNeurons, optimizer=optimizer, epochs=epochs, nrMinibatches=nrMinibatches, eta=eta, lmbda=lmbda, testSize=testSize, terminalUpdate=False)
            Creg.train()
            print(Creg)
            print("\n\n\n\n")
            accuracy[i,j] = Creg.finalTestLoss()
    idx = [r"{%.0f}"%k for k in layers]
    cols = [r"{%.0f}"%k for k in neurons]
    dF = pd.DataFrame(accuracy, index=idx, columns=cols)
    dF.to_pickle(output_path+filename+".pkl")
    
    info.set_file_info(filename+".pkl", method='SGD', opt=optimizer, no_epochs=epochs, no_minibatches=nrMinibatches, L=r"$[%s, %s]$" %(idx[0], idx[-1]), N=r"$[%s, %s]$" %(cols[0], cols[-1]), eta=eta, lmbda=lmbda, g=activationFunction, n_obs=Creg.n_obs, rho=rho)

def activationFunctionPerEpochAnalysis(filename):
    #   Fixed parameters
    hiddenLayers = 2    #Based on previous results
    neuronsInEachLayer=30    #Based on previous results
    eta = 1e-3  #Based on previous results
    lmbda = 1e-4    #Based on previous results
    outputNeurons=1
    epochs=[250, 1000]
    nrMinibatches=5
    testSize=0.2
    optimizer='RMSProp'
    rho = (0.9,0.999) # default hyperparams in RMSProp

    #   Parameters to test
    activationFunctions = ["sigmoid", "relu", "relu*", "tanh"]

    for epoch in epochs:
        dF = pd.DataFrame()
        for function in activationFunctions:
            Creg = CancerData(hiddenLayers=hiddenLayers, neuronsInEachLayer=neuronsInEachLayer, activationFunction=function, outputNeurons=outputNeurons, optimizer=optimizer, epochs=epoch, nrMinibatches=nrMinibatches, eta=eta, lmbda=lmbda, testSize=testSize, terminalUpdate=False)
            print(Creg)
            Creg.Net.train(extractInfoPerXEpoch=1)
            loss = np.asarray(Creg.Net.testLossPerEpoch)
            epochslist = np.asarray(Creg.Net.lossEpochs)
            dF[function] = loss
        dF["epochs"] = epochslist
        dF.to_pickle(output_path+filename+str(epoch)+".pkl")

        info.set_file_info(filename+str(epoch)+".pkl", method='SGD', opt=optimizer, no_epochs='...', no_minibatches=nrMinibatches, L=hiddenLayers, N=neuronsInEachLayer, eta=eta, lmbda=lmbda, g='...', n_obs=Creg.n_obs, rho=rho)

def EpochMinibatchAnalysis(filename):
    #   Fixed parameters
    hiddenLayers = 2    #Based on previous results
    neuronsInEachLayer=30   #Based on previous results
    eta = 1e-3  #Based on previous results
    lmbda = 1e-4    #Based on previous results
    outputNeurons=1
    activationFunction = 'tanh'
    testSize=0.2
    optimizer='RMSProp'
    rho = (0.9,0.999) # default hyperparams in RMSProp

    #   Parameters to test
    epoch_array = np.linspace(100, 1000, 10, dtype="int")
    minibatch_array = np.linspace(1, 10, 10, dtype="int")

    #  Parameters to find
    accuracy = np.zeros((len(epoch_array), len(minibatch_array)))

    #   Test loop
    for i, epoch in enumerate(epoch_array):
        for j, minibatch in enumerate(minibatch_array):
            print(f"Epoch: {epoch}\nMinibatch: {minibatch}")
            Creg = CancerData(hiddenLayers=hiddenLayers, neuronsInEachLayer=neuronsInEachLayer, activationFunction=activationFunction, outputNeurons=outputNeurons, optimizer=optimizer, epochs=epoch, nrMinibatches=minibatch, eta=eta, lmbda=lmbda, testSize=testSize, terminalUpdate=False)
            Creg.train()
            print(Creg)
            print("\n\n\n\n")
            accuracy[i,j] = Creg.finalTestLoss()
    idx = [r"{%.0f}"%k for k in epoch_array]
    cols = [r"{%.0f}"%k for k in minibatch_array]
    dF = pd.DataFrame(accuracy, index=idx, columns=cols)
    dF.to_pickle(output_path+filename+".pkl")

    info.set_file_info(filename+".pkl", method='SGD', opt=optimizer, no_epochs=r"$[%s, %s]$" %(idx[0], idx[-1]), no_minibatches=r"$[%s, %s]$" %(cols[0], cols[-1]), L=hiddenLayers, N=neuronsInEachLayer, eta=eta, lmbda=lmbda, g=activationFunction, n_obs=Creg.n_obs, rho=rho)

def logisticRegression(filename="logistic"):
    #   Fixed parameters
    hiddenLayers = 0
    outputNeurons=1
    activationFunction = 'sigmoid'
    epochs=250
    nrMinibatches=5
    testSize=0.2
    optimizer='adaGrad'
    rho = (0.9,0.999) # default hyperparams in RMSProp

    #   Parameters to test
    etas = np.logspace(-9,0,10)
    lmbdas = np.logspace(-9,0,10)

    #   Paramteres to find
    accuracy = np.zeros((len(lmbdas), len(etas)))

    #   Testing loop
    for i, lmbda in enumerate(lmbdas):
        for j, eta in enumerate(etas):
            print(f"Lmbda: {lmbda}\nEta: {eta}")
            Creg = CancerData(hiddenLayers=hiddenLayers, activationFunction=activationFunction, outputNeurons=outputNeurons, optimizer=optimizer, epochs=epochs, nrMinibatches=nrMinibatches, eta=eta, lmbda=lmbda, testSize=testSize, terminalUpdate=False)
            Creg.train()
            print(Creg)
            print("\n\n\n\n")
            accuracy[i,j] = Creg.finalTestLoss()
    idx = [r"$10^{%.0f}$"%k for k in np.log10(lmbdas)]
    cols = [r"$10^{%.0f}$"%k for k in np.log10(etas)]
    dF = pd.DataFrame(accuracy, index=idx, columns=cols)
    dF.to_pickle(output_path+filename+".pkl")

    info.set_file_info(filename+".pkl", method='SGD', opt=optimizer, no_epochs=epochs, no_minibatches=nrMinibatches, L=hiddenLayers, eta=r"$[%s, %s]$" %(cols[0], cols[-1]), lmbda=r"$[%s, %s]$" %(idx[0], idx[-1]), g=activationFunction, n_obs=Creg.n_obs, rho=rho)




if __name__=="__main__":
    try:
        log = sys.argv[1].strip().lower()
        if log in ['log', 'logistic']:
            logisticRegression()
        else:
            pass
    
    except IndexError:
        EtaLambdaAnalysis("EtaLmbdaMSECancer")
        # LayerNeuronsAnalysis("LayerNeuronCancer")
        # activationFunctionPerEpochAnalysis("actFuncPerEpochCancer")
        # EpochMinibatchAnalysis("EpochMinibatchCancer")
    


    # Update README.md and info.pkl
    info.additional_information("Loss function: cross entropy (with regularisation)")
    info.update(header="Results from cancer classification analysis using NN")
