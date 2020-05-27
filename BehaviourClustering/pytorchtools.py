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

class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))

def to_numpy(v):
    if is_list_or_tuple(v):
        return np.stack([to_numpy(sub_v) for sub_v in v], axis=1)
    try:
        return v.cpu().detach().numpy()

    except AttributeError:
        return v