import torch
import torch.nn as nn

numPatches = 10
numBounces = 2

# Create random Tensor to hold input
x = torch.randn(1, numPatches)
print(x)
print(torch.sum(x))

class FFNet(nn.Module):
    def __init__(self):
        super(FFNet, self).__init__()
        self.fc = nn.Linear(10, 10)
        self.fc.weight = torch.nn.Parameter(torch.ones_like(self.fc.weight))
        self.fc.bias = torch.nn.Parameter(torch.ones_like(self.fc.bias))

    def forward(self, x):
        with torch.no_grad():
            for bounces in range(0, numBounces):
                x = self.fc(x)
        return x


model = FFNet()

y_pred = model(x)

print(y_pred)
