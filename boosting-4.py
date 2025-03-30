from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor

from typing import Optional


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        plot: bool = False
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate

        self.plot = plot

        self.history = defaultdict(list) # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)  # Исправьте формулу на правильную - исправила. 
        self.feature_importances_: Optional[np.ndarray] = None

    def partial_fit(self, X_train, y_train, train_predictions, X_val=None, y_val=None, val_predictions=None):

        gradient = self.loss_derivative(y_train, train_predictions)
        model = self.base_model_class(**self.base_model_params)

        model.fit(X_train, -gradient)
        train_model_predictions = model.predict(X_train)

        gamma = self.find_optimal_gamma(y_train, train_predictions, train_model_predictions)
        self.models.append(model)
        self.gammas.append(gamma)

        train_predictions += self.learning_rate * gamma * train_model_predictions
        train_loss = self.loss_fn(y_train, train_predictions)
        
        val_loss = None
        if X_val is not None and y_val is not None:

            val_model_predictions = model.predict(X_val)
            val_predictions += self.learning_rate * gamma * val_model_predictions

            val_loss = self.loss_fn(y_val, val_predictions)

        return train_predictions, train_loss, val_predictions, val_loss
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, plot=False):
        """
        :param X_train: features array (train set)
        :param y_train: targets array (train set)
        :param X_val: features array (eval set)
        :param y_val: targets array (eval set)
        :param plot: bool 
        """
        train_predictions = np.zeros(y_train.shape[0])
        val_predictions = np.zeros(X_val.shape[0]) if X_val is not None else None

        for _ in range(self.n_estimators):
            train_predictions, train_loss, val_predictions, val_loss = self.partial_fit(
                X_train, y_train, train_predictions, X_val, y_val, val_predictions
            )
            self.history["train_loss"].append(train_loss)
            if val_loss is not None:
                self.history["val_loss"].append(val_loss)

        if self.plot:
            self.plot_history(X_train, y_train, X_val, y_val)
        
        self._compute_feature_importances()

    def predict_proba(self, X):
        predictions = np.zeros(X.shape[0])
        for model, gamma in zip(self.models, self.gammas):
            predictions += self.learning_rate * gamma * model.predict(X)
        return np.column_stack([1 - self.sigmoid(predictions), self.sigmoid(predictions)])

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, X, y):
        return score(self, X, y)
        
    def plot_history(self, X_train, y_train, X_val=None, y_val=None):
        """
        :param X: features array (any set)
        :param y: targets array (any set)
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(self.history["train_loss"], label="Train Loss")
        if X_val is not None and "val_loss" in self.history:
            plt.plot(self.history["val_loss"], label="Validation Loss")

        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid()
        plt.show()

    def _compute_feature_importances(self):
        
        if not self.models:
            self.feature_importances_ = None
            return

        importances = np.zeros_like(self.models[0].feature_importances_, dtype=float)

        for model in self.models:
            importances += model.feature_importances_

        total = importances.sum()
        if total > 0:
            importances /= total

        self.feature_importances_ = importances
