import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
def plot_grad_flow(named_parameters):
  '''Plots the gradients flowing through different layers in the net during training.
  Can be used for checking for possible gradient vanishing / exploding problems.
  
  Usage: Plug this function in Trainer class after loss.backwards() as 
  "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
  ave_grads = []
  max_grads= []
  layers = []
  for n, p in named_parameters:
      if(p.requires_grad) and ("bias" not in n):
          layers.append(n)
          ave_grads.append(p.grad.abs().mean())
          max_grads.append(p.grad.abs().max())
  plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
  plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
  plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
  plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
  plt.xlim(left=0, right=len(ave_grads))
  # plt.ylim(bottom = -0.001, top=0.2) # zoom in on the lower gradient regions
  plt.xlabel("Layers")
  plt.ylabel("average gradient")

def prepare_batch_rnn(batch,device):
  samples,labels=batch
  samples = samples.to(device).float()
  labels =torch.flatten(labels).to(device).long()

  return samples,labels

def prepare_batch_tsresnet(batch,device):
  samples,labels=batch
  samples=torch.transpose(samples,1,2)
  samples = samples.to(device).float()
  labels =torch.flatten(labels).to(device).long()

  return samples,labels
def init_weights(m):
    if type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif type(m) == nn.Conv1d or type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model,training_path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model,training_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model,training_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model,training_path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(),
                  os.path.join(training_path,"checkpoint.pth") )
        self.val_loss_min = val_loss