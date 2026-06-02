from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import numpy as np
from sklearn.metrics import accuracy_score


class TorchModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def fit(self, X, y):
        return self

    def predict(self, X):
        self.model.eval()

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            preds = outputs.argmax(dim=1)

        return preds.cpu().numpy()

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))