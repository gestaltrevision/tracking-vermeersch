import torch
import torch.nn as nn
import torch.nn.functional as F
import os 

class HAR_RNN(nn.Module):
    def __init__(self, hidden_size, num_layers,num_classes,
                    input_size=9,p_dropout=0.5,gpu_train=True):
        super(HAR_RNN, self).__init__()
        self.gpu_train=gpu_train
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=p_dropout)
        # self.bn1=nn.BatchNorm1d(hidden_size)
        self.fc1=nn.Linear(hidden_size,int(hidden_size/2))
        self.relu=nn.ReLU()  
    
        self.fc2=nn.Linear(int(hidden_size/2),num_classes)
        self.dropout = nn.Dropout(p=p_dropout)

        
    def forward(self, x):
        device = torch.device('cuda' if self.gpu_train else 'cpu')
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out=out[:,-1,:]
        out=self.dropout(self.relu(self.fc1(out)))
        out=self.fc2(out)

        return out


def _HarRnn(arch, hidden_size,layers, pretrained,models_dir, **kwargs):
    model =HAR_RNN(hidden_size, layers, **kwargs)

    if pretrained: 
        checkpoint_path=os.path.join(models_dir,arch,"checkpoint.pth")
        model.load_state_dict(torch.load(checkpoint_path))
    return model


def BaselineRnn(pretrained=False,models_dir="", **kwargs):
    r"""
    Args:
        pretrained (bool): If True, returns a model pre-trained on HAR dataset 
    """
    return _HarRnn('BaselineRnn', 128, 2, pretrained,
                        models_dir,**kwargs)