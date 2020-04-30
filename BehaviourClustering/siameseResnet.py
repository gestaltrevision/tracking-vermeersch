from TSResNet import TSResNet,BasicBlock
import torch
import os

class SiameseTSResNet(TSResNet):

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # x = self.fc(x)
        return x

    def forward(self, input1,input2):
        out1= self._forward_impl(input1)
        out2= self._forward_impl(input1)

        return [out1,out2]

def _tsresnet(arch, block, layers, pretrained,models_dir, **kwargs):
    model =SiameseTSResNet(block, layers, **kwargs)
    if pretrained: 
        checkpoint_path=os.path.join(models_dir,arch,"checkpoint.pth")
        model.load_state_dict(torch.load(checkpoint_path))

    return model


def tsresnet_shallow(pretrained=False,models_dir="", **kwargs):
    r"""
    Args:
        pretrained (bool): If True, returns a model pre-trained on HAR dataset 
    """
    return _tsresnet('tsresnet_shallow', BasicBlock, [1,1,1,1], pretrained,
                        models_dir,**kwargs)

def _prepare_sample(samples,device):
    samples=torch.transpose(samples,1,2)
    return samples.to(device).float()

def prepare_batch_siamese(batch,device):
    samples1,samples2, labels=batch
    samples1= _prepare_sample(samples1,device)
    samples2= _prepare_sample(samples2,device)
    labels =torch.flatten(labels).to(device).long()
    return samples1,samples2,labels

import torch.nn.functional as F
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, outputs, label):
    
        euclidean_distance = F.pairwise_distance(outputs[0], outputs[1], keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

if __name__ == "__main__":
    from BehaviourDatasets import TSDatasetSiamese
    import torch
    import torch.nn as nn
    import os
    import numpy as np
    from torch.utils.data import Dataset,DataLoader
    import joblib
    import itertools

    from transformers import LevelSelector
    from sklearn.preprocessing import LabelEncoder, RobustScaler


    scaler=RobustScaler()
    folder=r"C:\Users\jeuux\Desktop\Carrera\MoAI\TFM\AnnotatedData\Accelerometer_Data\Datasets\HAR_Dataset_raw" #Dataset without label grouping
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Hyper-parameters
    sequence_length = 50
    batch_size =128
    num_layers = 2
    data_types=[True,True,True] # select gaze
    level="Ds"
    n_components=9
    #creating train and valid datasets
    train_dataset= TSDatasetSiamese(folder,scaler,"Train",level,data_types)
    validation_dataset= TSDatasetSiamese(folder,scaler,"Val",level,data_types)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    val_loader= DataLoader(validation_dataset, batch_size=batch_size,shuffle=True)
    data_loaders=[train_loader,val_loader]

    batch=next(iter(train_loader))
    samples1,samples2,labels= prepare_batch_siamese(batch,device)
    
    print("Hi")
    #Init Model

    num_classes=train_dataset.num_classes
    model_arch=tsresnet_shallow
    model_params={"num_classes":num_classes,"n_components":n_components}
    prepare_batch_fcn=prepare_batch_tsresnet
    model = model_arch(**model_params).to(device)


    out=model(samples1,samples2)

    print("Ji again")

    criterion = ContrastiveLoss()
    loss_contrastive = criterion(out,labels)

    # loss_contrastive.backward()
    print("gud bie mai frend")

    pass

    