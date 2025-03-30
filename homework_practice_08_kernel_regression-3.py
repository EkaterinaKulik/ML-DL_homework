import numpy as np
from sklearn.base import RegressorMixin
from sklearn.gaussian_process.kernels import RBF


class KernelRidgeRegression(RegressorMixin):
    """
    Kernel Ridge regression class
    """

    def __init__(
        self,
        lr=0.01,
        regularization=1.0,
        tolerance=1e-2,
        max_iter=1000,
        batch_size=64,
        kernel_scale=1.0,
    ):
        """
        :param lr: learning rate
        :param regularization: regularization coefficient
        :param tolerance: stopping criterion for square of euclidean norm of weight difference
        :param max_iter: stopping criterion for iterations
        :param batch_size: size of the batches used in gradient descent steps
        :parame kernel_scale: length scale in RBF kernel formula
        """

        self.lr: float = lr
        self.regularization: float = regularization
        self.w: np.ndarray | None = None

        self.tolerance: float = tolerance
        self.max_iter: int = max_iter
        self.batch_size: int = batch_size
        self.loss_history: list[float] = []
        self.kernel = RBF(kernel_scale)

    def compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Calculating kernel matrix
        :param X1: first feature array
        :param X2: second feature array
        """
        return np.exp(-np.linalg.norm(X1[:, np.newaxis] - X2, axis=2) ** 2 / (2 * self.kernel.length_scale ** 2))

    def calc_loss(self, K: np.ndarray, y: np.ndarray) -> float:
        """
        Calculating loss for x and y dataset
        :param x: features array
        :param y: targets array
        """
        pred = K @ self.w
        loss = np.mean((pred - y) ** 2) + self.regularization * np.dot(self.w, self.w)
        return loss

    def calc_grad(self, K: np.ndarray, y: np.ndarray) -> float:
        """
        Calculating gradient for x and y dataset
        :param x: features array
        :param y: targets array
        """
        grad = K @ (K @ self.w + self.regularization * self.w - y)
        return grad

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KernelRidgeRegression":
        """
        Получение параметров с помощью градиентного спуска
        :param x: features array
        :param y: targets array
        :return: self
        """
        self.x_train = x
        K = self.compute_kernel_matrix(x, x)
        self.w = np.zeros(x.shape[0])  

        for i in range(self.max_iter):
            grad = self.calc_grad(K, y)
            w_new = self.w - self.lr * grad
            
            if np.linalg.norm(w_new - self.w, 2) < self.tolerance:
                break
            self.w = w_new

        return self

    def fit_closed_form(self, x: np.ndarray, y: np.ndarray) -> "KernelRidgeRegression":
        """
        Получение параметров через аналитическое решение
        :param x: features array
        :param y: targets array
        :return: self
        """
        self.x_train = x
        K = self.compute_kernel_matrix(x, x)
        I = np.eye(K.shape[0])
        self.w = np.linalg.solve(K + self.regularization * I, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicting targets for x dataset
        :param x: features array
        :return: prediction: np.ndarray
        """
        K_test = self.compute_kernel_matrix(x, self.x_train)
        return K_test @ self.w
