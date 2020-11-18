import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from tme5 import CirclesData

def init_params(nx, nh, ny):
    """
    nx, nh, ny: integers
    out params: dictionnary
    """
    params = {}

    # TODO remplir avec les paramètres Wh, Wy, bh, by
    # params["Wh"] = ...

    return params


def forward(params, X):
    """
    params: dictionnary
    X: (n_batch, dimension)
    """
    outputs = {}

    # TODO remplir avec les paramètres X, htilde, h, ytilde, yhat
    # outputs["X"] = ...

    return outputs['yhat'], outputs

def loss_accuracy(Yhat, Y):
    L = 0
    acc = 0

    # TODO

    return L, acc

def backward(params, outputs, Y):
    grads = {}

    # TODO remplir avec les paramètres Wy, Wh, by, bh
    # grads["Wy"] = ...

    return grads

def sgd(params, grads, eta):
    # TODO mettre à jour le contenu de params

    return params



if __name__ == '__main__':

    # init
    data = CirclesData()
    data.plot_data()
    N = data.Xtrain.shape[0]
    Nbatch = 10
    nx = data.Xtrain.shape[1]
    nh = 10
    ny = data.Ytrain.shape[1]
    eta = 0.03

    # Premiers tests, code à modifier
    params = init_params(nx, nh, xy)
    Yhat, outs = forward(params, data.Xtrain)
    L, _ = loss_accuracy(Yhat, Y)
    grads = backward(params, outputs, Y)
    params = sgd(params, grads, eta)

    # TODO apprentissage

    # attendre un appui sur une touche pour garder les figures
    input("done")
