import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np
from IMU_ut import plot_grad_flow


class HAR_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(HAR_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=0.75)
        self.bn1=nn.BatchNorm1d(hidden_size)
        self.fc1=nn.Linear(hidden_size,int(hidden_size/2))
        self.relu=nn.ReLU()  
        # self.bn2=nn.BatchNorm1d(int(hidden_size/2))
    
        self.fc2=nn.Linear(int(hidden_size/2),num_classes)
        self.dropout = nn.Dropout(p=0.75)

        
    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        # out=self.relu(out[:, -1, :])
        out=self.bn1(out[:, -1, :])
        out=self.dropout(self.relu(self.fc1(out)))
        out=self.fc2(out)

        out=F.log_softmax(out,dim=1)
        return out


