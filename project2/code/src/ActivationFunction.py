import numpy as np

class ActivationFunction:
    def __init__(self, activationFunction:str = "sigmoid") -> None:
        """Class to hold various activation functions and their derivatives.

        Args:
            activationFunction (str, optional): String with activation function name. Defaults to "sigmoid".
        """
        self.parseFunction(activationFunction)

    def parseFunction(self, stringToParse:str) -> None:
        """Sets the call function and derivative function of the class in accordance with the parsed string argument.

        Args:
            stringToParse (str): Activation function name to be parsed.

        Raises:
            ValueError: If parsed string is not captured by any activation function names.
        """
        g = stringToParse.strip().lower() # case insensitive
        if g in ['sigmoid', 'sigmiod']: # ...
            self.activationFunction = self.sigmoid
            self.derivativeFunction = self.sigmoidDerivative
        elif g in ['relu', 'rectified linear unit', 'rectifier']:
            self.activationFunction = self.ReLU
            self.derivativeFunction = self.ReLUDerivative
        elif g in ['leaky relu', 'lrelu', 'relu*', 'leaky rectifier']:
            self.activationFunction = self.leakyReLU
            self.derivativeFunction = self.leakyReLUDerivative
        elif g in ['tanh', 'hyperbolic tangent']:
            self.activationFunction = self.hyperbolicTangent
            self.derivativeFunction = self.hyperbolicTangentDerivative
        elif g in ['softmax']:
            self.activationFunction = self.softmax
            self.derivativeFunction = self.softmaxDerivative
        elif g in ['linear', 'lin']:
            self.activationFunction = self.linear
            self.derivativeFunction = self.linearDerivative
        else:
            raise ValueError(f"The library does not have functionalities for {g} activation function.")


    def derivative(self, a:np.ndarray) -> classmethod:
        """Return the correct derivative of the argument

        Args:
            a (np.ndarray): ndarray to be differentiated.

        Returns:
            function: parsed function.
        """
        return self.derivativeFunction(a)

    def __call__(self, a:np.ndarray) -> classmethod:
        """Return the correct function value of the argument.

        Args:
            a (np.ndarray): ndarray to find function value from.

        Returns:
            function: parsed function.
        """
        return self.activationFunction(a)

    def sigmoid(self, a:np.ndarray) -> np.ndarray:
        """Return sigmoid function value.

        Args:
            a (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return 1/(1+np.exp(-a))

    def sigmoidDerivative(self, a:np.ndarray) -> np.ndarray:
        """Return derivative of sigmoid function value.

        Args:
            a (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return (self.sigmoid(a) * (1-self.sigmoid(a)))

    def ReLU(self, a:np.ndarray) -> np.ndarray:
        """Return RELU function value.

        Args:
            a (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return np.where(a > 0, a, 0)

    def ReLUDerivative(self, a:np.ndarray) -> np.ndarray:
        """Return derivative of RELU function value.

        Args:
            a (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return np.where(a > 0, 1, 0)
    
    def leakyReLU(self, a:np.ndarray) -> np.ndarray:
        """Return leaky RELU function value.

        Args:
            a (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        # return self.leakyReLUDerivative(a)*a
        return np.where(a>0, a, 0.1*a)

    def leakyReLUDerivative(self, a:np.ndarray) -> np.ndarray:
        """Return derivative of leaky RELU function value.

        Args:
            a (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return np.where(a>0, 1, 0.01)

    def hyperbolicTangent(self, a:np.ndarray) -> np.ndarray:
        """Return hyperbolic tangent function value.

        Args:
            a (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return np.tanh(a)
    
    def hyperbolicTangentDerivative(self, a:np.ndarray) -> np.ndarray:
        """Return derivative of hyperbolic tangent function value.

        Args:
            a (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return 1 - np.tanh(a)**2

    # def softmax(self, a):
    #     exps = np.exp(a)
    #     return exps/np.sum(exps)

    # def softmaxDerivative(self, a):
    #     sm = self.softmax(a)

    #     # do not think this actually works...
    #     # https://e2eml.school/softmax.html
    #     return sm*np.identity(sm.size) - sm.transpose() @ sm

    def linear(self, a:np.ndarray) -> np.ndarray:
        """Return linear function value.

        Args:
            a (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return a
        # return a

    def linearDerivative(self, a:np.ndarray) -> np.ndarray:
        """Return derivative of linear function value.

        Args:
            a (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return 1

if __name__=="__main__":
    pass