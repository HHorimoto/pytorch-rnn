import torch

import numpy as np
import matplotlib.pyplot as plt

def plot(results: dict, metric: str):
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel(metric)
    for key, value in results.items():
        plt.plot(value, label=key)
    plt.legend()
    plt.savefig(metric+'.png')

def plot_preds(net, dataloader, dataset, n_hidden, device, rnn_name):
    net.eval()
    prediction_result = []

    hx = torch.zeros(1, n_hidden).to(device)
    cx = torch.zeros(1, n_hidden).to(device)

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            output, hx, cx = net(X[:, 0, :], hx, cx)

            prediction_result.append(output.item())

    prediction_result = np.array(prediction_result).flatten()

    plt.figure()
    plt.title(rnn_name)
    plt.plot(dataset.label, color='red', label='true')
    plt.plot(prediction_result.tolist(), color='blue', label='pred')
    plt.legend()
    plt.savefig('preds.png')