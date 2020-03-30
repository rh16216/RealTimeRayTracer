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

    def calculatePatchVertices(self, vertex):
        if (self.stepX == 0):
            vertex1 = vertex + torch.tensor([0.0, 0.0, self.stepZ])
            vertex2 = vertex1 + torch.tensor([0.0, self.stepY, 0.0])
            vertex3 = vertex2 - torch.tensor([0.0, 0.0, self.stepZ])

        if (self.stepY == 0):
            vertex1 = vertex + torch.tensor([self.stepX, 0.0, 0.0])
            vertex2 = vertex1 + torch.tensor([0.0, 0.0, self.stepZ])
            vertex3 = vertex2 - torch.tensor([self.stepX, 0.0, 0.0])

        if (self.stepZ == 0):
            vertex1 = vertex + torch.tensor([self.stepX, 0.0, 0.0 ])
            vertex2 = vertex1 + torch.tensor([0.0, self.stepY, 0.0])
            vertex3 = vertex2 - torch.tensor([self.stepX, 0.0, 0.0])

        return vertex1, vertex2, vertex3


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

def drawLine(vertex0X, vertex0Y, vertex1X, vertex1Y, colour, screen):
     xDiff = vertex1X - vertex0X
     yDiff = vertex1Y - vertex0Y
     numSteps = max(abs(xDiff), abs(yDiff))

     if (numSteps == 0):
         screen[round(vertex0Y)][round(vertex0X)] = colour
     else:
         xStep = xDiff/numSteps
         yStep = yDiff/numSteps

         for step in range(0, numSteps+1):
             screen[round(vertex0Y + yStep*step)][round(vertex0X + xStep*step)] = colour


def calculateLine(vertex0X, vertex0Y, vertex1X, vertex1Y):
    xDiff = vertex1X - vertex0X
    yDiff = vertex1Y - vertex0Y
    numSteps = max(abs(xDiff), abs(yDiff))

    xStep = xDiff/numSteps
    yStep = yDiff/numSteps

    lineList = []
    for step in range(0, numSteps+1):
        lineList.append((round(vertex0Y + yStep*step), round(vertex0X + xStep*step)))

    return lineList

def fillPatch(line0, line1, line2, line3, colour, screen):
    maxY = int(max(line0[0][1], line1[0][1], line2[0][1], line3[0][1]))
    minY = int(min(line0[0][1], line1[0][1], line2[0][1], line3[0][1]))
    yDiff = (maxY - minY)+1

    maxXs = [-math.inf]*yDiff
    minXs = [math.inf]*yDiff

    for pixel in line0:
        if (pixel[0] < minXs[pixel[1]-minY]):
            minXs[pixel[1]-minY] = pixel[0]
        if (pixel[0] > maxXs[pixel[1]-minY]):
            maxXs[pixel[1]-minY] = pixel[0]

    for pixel in line1:
        if (pixel[0] < minXs[pixel[1]-minY]):
            minXs[pixel[1]-minY] = pixel[0]
        if (pixel[0] > maxXs[pixel[1]-minY]):
            maxXs[pixel[1]-minY] = pixel[0]

    for pixel in line2:
        if (pixel[0] < minXs[pixel[1]-minY]):
            minXs[pixel[1]-minY] = pixel[0]
        if (pixel[0] > maxXs[pixel[1]-minY]):
            maxXs[pixel[1]-minY] = pixel[0]

    for pixel in line3:
        if (pixel[0] < minXs[pixel[1]-minY]):
            minXs[pixel[1]-minY] = pixel[0]
        if (pixel[0] > maxXs[pixel[1]-minY]):
            maxXs[pixel[1]-minY] = pixel[0]


    for index in range(0, yDiff):
        drawLine(minXs[index], minY+index, maxXs[index], minY+index, colour, screen)

def projectVertex(vertex, focalLength, width, height):
    projectedVertex = vertex*focalLength/vertex[2] + torch.tensor([width/2, height/2, 0.0])

    #negating due to swap in direction of X and Y
    projectedVertexX = round(-1.0*projectedVertex[0].item())
    projectedVertexY = round(-1.0*projectedVertex[1].item())

    return projectedVertexX, projectedVertexY



def projectToScreen(meshList, width, height):

    focalLength = height/2
    cameraPos = torch.tensor([278.0, 273.0, -600.0])

    screen = torch.zeros(height, width, 3)

    for mesh in meshList:
        for index, vertex in enumerate(mesh.meshGrid):
            vertex0 = vertex - cameraPos
            vertex1, vertex2, vertex3 = mesh.calculatePatchVertices(vertex0)

            projectedVertex0X, projectedVertex0Y = projectVertex(vertex0, focalLength, width, height)
            projectedVertex1X, projectedVertex1Y = projectVertex(vertex1, focalLength, width, height)
            projectedVertex2X, projectedVertex2Y = projectVertex(vertex2, focalLength, width, height)
            projectedVertex3X, projectedVertex3Y = projectVertex(vertex3, focalLength, width, height)

            line0 = calculateLine(projectedVertex0X, projectedVertex0Y, projectedVertex1X, projectedVertex1Y)
            line1 = calculateLine(projectedVertex1X, projectedVertex1Y, projectedVertex2X, projectedVertex2Y)
            line2 = calculateLine(projectedVertex2X, projectedVertex2Y, projectedVertex3X, projectedVertex3Y)
            line3 = calculateLine(projectedVertex3X, projectedVertex3Y, projectedVertex0X, projectedVertex0Y)

            fillPatch(line0, line1, line2, line3, torch.tensor([0.0, 0.0, 50.0*(index%6)]), screen)
            #screen[projectedVertex1Y][projectedVertex1X] = torch.tensor([255.0, 255.0, 255.0])
            #screen[projectedVertex2Y][projectedVertex2X] = torch.tensor([255.0, 0.0, 0.0])
            #screen[projectedVertex3Y][projectedVertex3X] = torch.tensor([0.0, 0.0, 255.0])

    return screen

def writePPM(data, width, height, fileName):
    f = open(fileName, "w+")

    f.write("P3 \n")
    f.write(str(width) + " " + str(height) + "\n")
    f.write("255 \n")

    for y in range(0, height):
        for x in range(0, width):
            f.write(str(int(data[y][x][0].item())) + " ")
            f.write(str(int(data[y][x][1].item())) + " ")
            f.write(str(int(data[y][x][2].item())) + " ")
            f.write(" ")
        f.write("\n")

    f.close()


numPatches = 500
numBounces = 2

# Create random Tensor to hold input
x = torch.randn(1, numPatches)
#print(x)
#print(torch.sum(x))

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

#print(time.perf_counter())
#ffGrid = calculateFFGrid(meshList) #TODO: Pickle this and load in values
#print(time.perf_counter())
#print(ffGrid)

#model = FFNet(ffGrid)

#print(time.perf_counter())
#y_pred = model(x)
#print(time.perf_counter())
#print(y_pred)

width = 512
height = 512

data = projectToScreen(meshList, width, height)

writePPM(data, width, height, "output.ppm")
