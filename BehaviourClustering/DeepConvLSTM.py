import torch
import torch.nn as nn
import torch.nn.functional as F
import os 

class DeepConvLSTM(nn.Module):
    
    def __init__(self,n_filters,num_classes,filter_size=3,num_layers=2,hidden_size=128,
                n_channels=9,n_classes=2,p_dropout=0.5,gpu_train=True):
        super(DeepConvLSTM, self).__init__()
        self.gpu_train=gpu_train
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.n_channels=n_channels
        self.num_layers=num_layers
        self.hidden_size=hidden_size
        self.n_classes=n_classes
        
        #Conv Filters
        self.conv1 = nn.Conv1d(n_channels, n_filters, filter_size)
        self.conv2 = nn.Conv1d(n_filters, n_filters * 2, filter_size)
        self.conv3 = nn.Conv1d(n_filters*2, n_filters * 4, filter_size)
        self.conv4 = nn.Conv1d(n_filters*4, n_filters * 8, filter_size)

        #Temporal Path
        self.lstm = nn.LSTM(n_filters * 8 , hidden_size, num_layers, batch_first=True)
        self.fc1=nn.Linear(hidden_size,int(hidden_size/2))
        self.relu=nn.ReLU()  
        self.fc2=nn.Linear(int(hidden_size/2),num_classes)
        self.dropout = nn.Dropout(p=p_dropout)


    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x)) #out of conv layers (batch_size,n_filters,seq_length)

        x=torch.transpose(x,1,2) #input to lstm layers (batch_size,seq_length,n_filters)

        # Set initial hidden and cell states 
        device = torch.device('cuda' if self.gpu_train else 'cpu')
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of sh

        # Decode the hidden state of the last time step
        out=out[:,-1,:]
        out=self.dropout(self.relu(self.fc1(out)))
        out=self.fc2(out)
        
        return out


def _DeepConvLSTM(arch, n_filters, pretrained, models_dir, **kwargs):
    model =DeepConvLSTM(n_filters,**kwargs)

    if pretrained: 
        checkpoint_path=os.path.join(models_dir,arch,"checkpoint.pth")
        model.load_state_dict(torch.load(checkpoint_path))
    return model


def DeepConvLSTM_8(pretrained=False,models_dir="", **kwargs):
    r"""
    Args:
        pretrained (bool): If True, returns a model pre-trained on HAR dataset 
    """
    return _DeepConvLSTM('DeepConvLSTM_8', 8, pretrained,
                        models_dir,**kwargs)

def DeepConvLSTM_16(pretrained=False,models_dir="", **kwargs):
    r"""
    Args:
        pretrained (bool): If True, returns a model pre-trained on HAR dataset 
    """
    return _DeepConvLSTM('DeepConvLSTM_16', 16, pretrained,
                        models_dir,**kwargs)