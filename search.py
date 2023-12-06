from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier
from AE_Classifier_Model import Autoencoder  # replace with your model class
from AE_Classifier_Train import train_autoencoder  # replace with your training function
from Loss import CosineLoss

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define the parameter grid for the grid search
param_grid = {
    'lr': [0.01, 0.001, 0.0001],
    'max_epochs': [10, 20, 30],
    # add more parameters here if needed
}

# Create a Skorch classifier with your PyTorch model
net = NeuralNetClassifier(
    train_autoencoder(),
    max_epochs=1000,
    lr=0.01,
    # Add your PyTorch model parameters here
)

net = NeuralNetRegressor(
    Autoencoder,
    criterion=CosineLoss,
    max_epochs=10,
    optimizer=optim.Adam,
    optimizer__lr = .005
)

# Perform the grid search
gs = GridSearchCV(net, param_grid, refit=False, cv=3, scoring='accuracy')
gs.fit(X, y)  # replace X, y with your data

print(gs.best_score_, gs.best_params_)