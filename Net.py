import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

    
class Student_resnet18(nn.Module):
    def __init__(self,first_conv,class_num,pretrained=True):
        super(Student_resnet18, self).__init__()
       
        self.resnet18 = models.resnet18(pretrained=pretrained)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 128)
        if(first_conv==False):
            self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
            self.resnet18.maxpool = nn.Identity()
            
        self.projector = nn.Sequential(nn.Linear(128, 128, bias=False), nn.BatchNorm1d(128),
                               nn.ReLU(inplace=True), nn.Linear(128, class_num, bias=False))
       
    def forward(self, x, return_feature=False, return_embedding=False):
        """Run ResNet18 and optionally return the layer4 feature map.

        Args:
            x: input tensor [B, C, H, W].
            return_feature: if True, return (logits, layer4_feature_map).
            return_embedding: if True, return (embedding, logits) for legacy use.

        Returns:
            logits by default; optional tuples described above.
        """
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        feature_map = self.resnet18.layer4(x)

        features = self.resnet18.avgpool(feature_map)
        features = torch.flatten(features, 1)
        features = self.resnet18.fc(features)
        y = self.projector(features)

        if return_feature:
            return y, feature_map
        if return_embedding:
            return features, y
        return y   
    


