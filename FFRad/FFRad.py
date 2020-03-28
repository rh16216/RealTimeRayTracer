import torch
import torch.nn as nn

# Floor  -- white lambert
#{    0.0,    0.0,    0.0, 0.0 },
#{    0.0,    0.0,  559.2, 0.0 },
#{  556.0,    0.0,  559.2, 0.0 },
#{  556.0,    0.0,    0.0, 0.0 },

# Ceiling -- white lambert
#{    0.0,  548.8,    0.0,  0.0 },
#{  556.0,  548.8,    0.0,  0.0 },
#{  556.0,  548.8,  559.2,  0.0 },
#{    0.0,  548.8,  559.2,  0.0 },

# Back wall -- white lambert
#{    0.0,    0.0,  559.2,  0.0 },
#{    0.0,  548.8,  559.2,  0.0 },
#{  556.0,  548.8,  559.2,  0.0 },
#{  556.0,    0.0,  559.2,  0.0 },

# Right wall -- green lambert
#{    0.0,    0.0,    0.0,  0.0 },
#{    0.0,  548.8,    0.0,  0.0 },
#{    0.0,  548.8,  559.2,  0.0 },
#{    0.0,    0.0,  559.2,  0.0 },

# Left wall -- red lambert
#{  556.0,    0.0,    0.0,  0.0 },
#{  556.0,    0.0,  559.2,  0.0 },
#{  556.0,  548.8,  559.2,  0.0 },
#{  556.0,  548.8,    0.0,  0.0 },

def createBasicMesh(minX, maxX, minY, maxY, minZ, maxZ, numDivides):
    ignoreX = (minX == maxX)
    ignoreY = (minY == maxY)
    ignoreZ = (minZ == maxZ)

    if (ignoreX):
        dimX = torch.tensor([0.0])
    else:
        dimX = torch.arange(minX, maxX+1.0, (maxX-minX)/(numDivides-1.0))

    if (ignoreY):
        dimY = torch.tensor([0.0])
    else:
        dimY = torch.arange(minY, maxY+1.0, (maxY-minY)/(numDivides-1.0))

    if (ignoreZ):
        dimZ = torch.tensor([0.0])
    else:
        dimZ = torch.arange(minZ, maxZ+1.0, (maxZ-minZ)/(numDivides-1.0))

    mesh = torch.cartesian_prod(dimX, dimY, dimZ)
    print(mesh.size())

    return mesh

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

floor     = createBasicMesh(0.0, 556.0, 0.0, 0.0, 0.0, 559.2, 10)
ceiling   = createBasicMesh(0.0, 556.0, 548.8, 548.8, 0.0, 559.2, 10)
backWall  = createBasicMesh(0.0, 556.0, 0.0, 548.8, 559.2, 559.2, 10)
rightWall = createBasicMesh(0.0, 0.0, 0.0, 548.8, 0.0, 559.2, 10)
leftWall  = createBasicMesh(556.0, 556.0, 0.0, 548.8, 0.0, 559.2, 10)

print(y_pred)
