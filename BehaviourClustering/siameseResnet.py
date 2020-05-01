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

    def forward(self, inputs):
        out1= self._forward_impl(inputs[0])
        out2= self._forward_impl(inputs[1])

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

