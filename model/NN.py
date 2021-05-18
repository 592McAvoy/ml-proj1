'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''

'''
All standard network architectures for CIFAR-10 images (also
32x32 pixels) can be applied to this project.

You may read papers and do a survey on the network architecture, but
you do NOT need to try all architectures. 
Just learn >=2 DNNs
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

# __all__ = ['alexnet']

class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, k_size, stride=1, padding=0, pooling=False):
        super(BasicBlock, self).__init__()
        components = [
            nn.Conv2d(in_c, out_c, kernel_size=k_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True)
        ]
        if pooling:            
            components.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.components = nn.Sequential(*components)
        
    def forward(self, x):
        # print(x.size())
        return self.components(x)

class BasicBNBlock(nn.Module):
    def __init__(self, in_c, out_c, k_size, stride=1, padding=0, pooling=False):
        super(BasicBNBlock, self).__init__()
        components = [
            nn.Conv2d(in_c, out_c, kernel_size=k_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_c)
        ]
        if pooling:            
            components.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.components = nn.Sequential(*components)
        
    def forward(self, x):
        # print(x.size())
        return self.components(x)
    
class DeepNeuralNetwork(BaseModel):
    """
    AlexNet based DNN
    """
    def __init__(self, num_classes=10, block_type='basic'):
        super().__init__()
        if block_type == 'basic':
            block = BasicBlock
        if block_type == 'basic_bn':
            block = BasicBNBlock
        else:
            raise NotImplementedError(block_type)

        self.feature_ext = nn.Sequential(
            block(in_c=3, out_c=64, k_size=11, stride=4, padding=5, pooling=True),
            block(in_c=64, out_c=192, k_size=5, padding=2, pooling=True),
            block(in_c=192, out_c=384, k_size=3, padding=1),
            block(in_c=384, out_c=256, k_size=3, padding=1),
            block(in_c=256, out_c=256, k_size=3, padding=1, pooling=True),            
        )
        self.fc = nn.Linear(256, num_classes)

    def features(self, x):
        x = self.feature_ext(x)
        x = x.view(x.size(0), -1)
        return x
    
    def classifier(self, fea):
        return self.fc(fea)

    def forward(self, x):
        x = self.features(x)        
        x = self.classifier(x)
        return x

# class AlexNet(BaseModel):
    
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.classifier = nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x


# def alexnet(**kwargs):
#     r"""AlexNet model architecture from the
#     `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
#     """
#     model = AlexNet(**kwargs)
#     return model

if __name__ == '__main__':
    nn = DeepNeuralNetwork()
    print(nn)
    dummy = torch.rand((10,3, 32, 32))
    nn(dummy)
    