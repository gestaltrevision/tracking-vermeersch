import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np
from IMU_ut import plot_grad_flow

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=0.8)
                            
        self.relu=nn.ReLU()
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        out=F.softmax(out,dim=1)
        return out

class Trainer(object):
  def __init__(self,model,criterion,optimizer,device):
    self.model=model
    self.criterion=criterion
    self.optimizer=optimizer
    self.device=device

