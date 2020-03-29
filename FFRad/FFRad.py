import torch
import torch.nn as nn

import math
import time
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

class Mesh:
    def __init__(self, stepX, stepY, stepZ, meshGrid, centreGrid):
        self.stepX = stepX
        self.stepY = stepY
        self.stepZ = stepZ
        self.meshGrid = meshGrid
        self.centreGrid = centreGrid

    def area(self):
        dimX = self.stepX
        dimY = self.stepY
        dimZ = self.stepZ

        if (dimX == 0.0):
            dimX = 1.0
        if (dimY == 0.0):
            dimY = 1.0
        if (dimZ == 0.0):
            dimZ = 1.0

        return dimX*dimY*dimZ

    def norm(self):
        norm = torch.tensor([0.0, 0.0, 0.0])
        if (self.stepX == 0.0): norm = torch.tensor([1.0, 0.0, 0.0])
        if (self.stepY == 0.0): norm = torch.tensor([0.0, 1.0, 0.0])
        if (self.stepZ == 0.0): norm = torch.tensor([0.0, 0.0, 1.0])

        centre = torch.tensor([278.0, 274.4, 279.6])
        centreDot = -1.0*(norm[0]*centre[0] + norm[1]*centre[1] + norm[2]*centre[2])
        if (centreDot < 0): norm = -1.0*norm

        return norm


def createBasicMesh(minX, maxX, minY, maxY, minZ, maxZ, numDivides):
    ignoreX = (minX == maxX)
    ignoreY = (minY == maxY)
    ignoreZ = (minZ == maxZ)

    dimX = torch.tensor([minX])
    dimY = torch.tensor([minY])
    dimZ = torch.tensor([minZ])

    stepX = 0.0
    stepY = 0.0
    stepZ = 0.0

    if (not ignoreX):
        stepX = (maxX-minX)/numDivides
        dimX = torch.arange(minX, maxX, stepX)

    if (not ignoreY):
        stepY = (maxY-minY)/numDivides
        dimY = torch.arange(minY, maxY, stepY)

    if (not ignoreZ):
        stepZ = (maxZ-minZ)/numDivides
        dimZ = torch.arange(minZ, maxZ, stepZ)

    meshGrid = torch.cartesian_prod(dimX, dimY, dimZ)
    #print(meshGrid.size())

    centreGrid = torch.add(meshGrid, torch.tensor([stepX/2.0, stepY/2.0, stepZ/2.0]))
    #print(centreGrid.size())

    mesh = Mesh(stepX, stepY, stepZ, meshGrid, centreGrid)

    return mesh


def calculateFFGridPair(mesh1, mesh2):
    ffGrid = torch.zeros(100, 100) #TODO: calculate from input mesh

    for index1, centre1 in enumerate(mesh1.centreGrid):
        for index2, centre2 in enumerate(mesh2.centreGrid):
            diff = centre1 - centre2
            diffLength = math.sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2])
            diffDir = diff/diffLength
            centre1Norm = mesh1.norm()
            centre2Norm = mesh2.norm()
            dot1 = -1.0*(centre1Norm[0]*diffDir[0] + centre1Norm[1]*diffDir[1] + centre1Norm[2]*diffDir[2])
            dot2 = centre2Norm[0]*diffDir[0] + centre2Norm[1]*diffDir[1] + centre2Norm[2]*diffDir[2]
            area2 = mesh2.area()

            ffGrid[index1][index2] = dot1*dot2*area2/(3.14*diffLength*diffLength)


    return ffGrid

def calculateFFGrid(meshList):
    ffGridPair = torch.zeros(100, 100) #TODO: calculate from input mesh
    ffGrid = torch.zeros(500, 500) #TODO: calculate from input mesh
    for index1, mesh1 in enumerate(meshList):
        for index2, mesh2 in enumerate(meshList):
            if (index1 == index2):
                ffGridPair = torch.zeros(100, 100) #TODO: calculate from input mesh
            else:
                ffGridPair = calculateFFGridPair(mesh1, mesh2)

            for i in range(0, 100):
                for j in range(0, 100):
                    ffGrid[100*index1+i][100*index2+j] = ffGridPair[i][j]


    return ffGrid


numPatches = 500
numBounces = 2

# Create random Tensor to hold input
x = torch.randn(1, numPatches)
print(x)
print(torch.sum(x))

class FFNet(nn.Module):
    def __init__(self, ffGrid):
        super(FFNet, self).__init__()
        self.fc = nn.Linear(numPatches, numPatches)
        #self.fc.weight = torch.nn.Parameter(torch.ones_like(self.fc.weight))
        self.fc.weight = torch.nn.Parameter(ffGrid)
        self.fc.bias = torch.nn.Parameter(torch.ones_like(self.fc.bias))

    def forward(self, x):
        with torch.no_grad():
            for bounces in range(0, numBounces):
                x = self.fc(x)
        return x


floorMesh     = createBasicMesh(0.0, 556.0, 0.0, 0.0, 0.0, 559.2, 10)
ceilingMesh   = createBasicMesh(0.0, 556.0, 548.8, 548.8, 0.0, 559.2, 10)
backWallMesh  = createBasicMesh(0.0, 556.0, 0.0, 548.8, 559.2, 559.2, 10)
rightWallMesh = createBasicMesh(0.0, 0.0, 0.0, 548.8, 0.0, 559.2, 10)
leftWallMesh  = createBasicMesh(556.0, 556.0, 0.0, 548.8, 0.0, 559.2, 10)

meshList = [floorMesh, ceilingMesh, backWallMesh, rightWallMesh, leftWallMesh]

ffGrid = calculateFFGrid(meshList)
#print(ffGrid)

model = FFNet(ffGrid)

print(time.perf_counter())
y_pred = model(x)
print(time.perf_counter())
print(y_pred)
