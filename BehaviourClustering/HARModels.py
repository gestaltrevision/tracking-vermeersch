import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np
from pytorchtools import plot_grad_flow


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


class DeepConvLSTM(nn.Module):
    
    def __init__(self,n_filters,filter_size,hidden_size,num_layers=2,n_channels=6,n_classes=2,p_dropout=0.5):
        super(DeepConvLSTM, self).__init__()
        self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.n_channels=n_channels
        self.num_layers=num_layers
        self.hidden_size=hidden_size
        self.n_classes=n_classes
             
        self.conv1 = nn.Conv1d(n_channels, n_filters, filter_size)
        self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.conv4 = nn.Conv1d(n_filters, n_filters, filter_size)

        self.lstm = nn.LSTM(n_filters, hidden_size, num_layers, batch_first=True)
        # self.bn1=nn.BatchNorm1d(hidden_size)
        self.fc=nn.Linear(hidden_size,n_classes)
        # self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x)) #out of conv layers (batch_size,n_filters,seq_length)

        x=torch.transpose(x,1,2) #input to lstm layers (batch_size,seq_length,n_filters)

        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of sh

        # Decode the hidden state of the last time step
        out=out[:,-1,:]
        # out=self.bn1(out)
        # out=self.dropout(F.relu(self.fc(out)))
        out=self.fc(out)
        out=F.log_softmax(out,dim=1)
        
        return out