from util_libs import *


#####################################################
# Neural net architectures
####################################################

class FedNet(nn.Module):
    def __init__(self,n_c):
        super(FedNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*5*5, 384)
        self.fc2 = nn.Linear(384,192)
        self.fc3 = nn.Linear(192, n_c)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*5*5)
        x = F.relu(self.fc1(x))
        features = F.relu(self.fc2(x))   # ← penultimate representation
        logits   = self.fc3(features)
        return features, logits


class EMNIST_CNN(nn.Module):
    def __init__(self):
        super(EMNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(12*12*64, 128)
        self.fc2 = nn.Linear(128, 62)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)


    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = x.view(x.size(0), -1)
        features = func.relu(self.fc1(x))   # ← penultimate representation
        features = self.dropout2(features)
        logits   = self.fc2(features)
        return features, logits
    
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(2,planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(2,planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(2,self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(2,64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 3)
        features = out.view(out.size(0), -1)   # ← penultimate representation (512-d)
        logits   = self.linear(features)
        return features, logits
def resnet18(n_c):
    return ResNet(BasicBlock, [2,2,2,2],num_classes=n_c)


def get_model(model,n_c):
  if(model=='resnet18'):
    return resnet18(n_c)
  elif(model == 'CNN'):
    return FedNet(n_c)
  elif(model=='EMNIST_CNN'):
    return EMNIST_CNN()

def get_feature_dim(model_name):
    """Return the penultimate-layer dimension for TempNet construction."""
    if model_name == 'resnet18':
        return 512
    elif model_name == 'CNN':
        return 192
    elif model_name == 'EMNIST_CNN':
        return 128
    else:
        raise ValueError(f"Unknown model: {model_name}")

class TempNet(nn.Module):
    """Temperature Network for adaptive logit scaling"""
    def __init__(self, feature_dim=512, hidden_dim=128, tau_min=0.05, tau_max=2.0):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.tau_min = tau_min
        self.tau_max = tau_max

    def forward(self, x):
        h = F.relu(self.fc1(x))
        raw = self.fc2(h)
        tau = torch.sigmoid(raw)
        tau = tau * (self.tau_max - self.tau_min) + self.tau_min
        return tau.mean()