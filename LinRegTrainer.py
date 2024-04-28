import numpy as np
from numpy.typing import NDArray


class LinRegTrainer:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], X: NDArray[np.float64]) -> np.ndarray:
        # calculate the derivative using the squared loss function
        N = len(X)
        return -2 * np.dot(ground_truth - model_prediction, X) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # squeeze is optional in this case, but it is used to remove the extra dimension
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self, 
        X: NDArray[np.float64], 
        Y: NDArray[np.float64], 
        num_iterations: int, 
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # train the model using the gradient descent algorithm and update according to learning rate
        for i in range(num_iterations):
            preds = self.get_model_prediction(X, initial_weights)
            grad = self.get_derivative(preds, Y, X)
            initial_weights -= self.learning_rate * grad

        return np.round(initial_weights, 5)