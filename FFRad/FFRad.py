import torch
import torch.nn as nn

import math
import time
import pickle

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

    def norm(self, vertex):
        norm = torch.tensor([0.0, 0.0, 0.0])
        if (self.stepX == 0.0): norm = torch.tensor([1.0, 0.0, 0.0])
        if (self.stepY == 0.0): norm = torch.tensor([0.0, 1.0, 0.0])
        if (self.stepZ == 0.0): norm = torch.tensor([0.0, 0.0, 1.0])

        centre = torch.tensor([278.0, 274.4, 279.6]) - vertex
        centreDot = norm[0]*centre[0] + norm[1]*centre[1] + norm[2]*centre[2]
        if (centreDot < 0):
            norm = -1.0*norm

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
        dimX = torch.arange(minX, maxX+1.0, stepX)

    if (not ignoreY):
        stepY = (maxY-minY)/numDivides
        dimY = torch.arange(minY, maxY+1.0, stepY)

    if (not ignoreZ):
        stepZ = (maxZ-minZ)/numDivides
        dimZ = torch.arange(minZ, maxZ+1.0, stepZ)

    meshGrid = torch.cartesian_prod(dimX, dimY, dimZ)
    #print(meshGrid.size())

    centreGrid = torch.add(meshGrid, torch.tensor([stepX/2.0, stepY/2.0, stepZ/2.0]))
    #print(centreGrid.size())

    mesh = Mesh(stepX, stepY, stepZ, meshGrid, centreGrid)

    return mesh


def calculateFFGridPair(mesh1, mesh2):
    ffGrid = torch.zeros(121, 121) #TODO: calculate from input mesh

    for index1, vertex1 in enumerate(mesh1.meshGrid):
        for index2, vertex2 in enumerate(mesh2.meshGrid):
            diff = vertex1 - vertex2
            diffLength = math.sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2])
            if (diffLength == 0.0):
                ffGrid[index1][index2] = 0.0

            else:

                diffDir = diff/diffLength
                vertex1Norm = mesh1.norm(vertex1)
                vertex2Norm = mesh2.norm(vertex2)
                dot1 = -1.0*(vertex1Norm[0]*diffDir[0] + vertex1Norm[1]*diffDir[1] + vertex1Norm[2]*diffDir[2])
                dot1 = max(dot1, 0.0)
                dot2 = vertex2Norm[0]*diffDir[0] + vertex2Norm[1]*diffDir[1] + vertex2Norm[2]*diffDir[2]
                dot2 = max(dot2, 0.0)
                area2 = mesh2.area()

                ffGrid[index1][index2] = dot1*dot2*area2/(3.14*diffLength*diffLength)

    return ffGrid

def calculateFFGrid(meshList):
    ffGridPair = torch.zeros(121, 121) #TODO: calculate from input mesh
    ffGrid = torch.zeros(605, 605) #TODO: calculate from input mesh
    for index1, mesh1 in enumerate(meshList):
        for index2, mesh2 in enumerate(meshList):
            if (index1 == index2):
                ffGridPair = torch.zeros(121, 121) #TODO: calculate from input mesh
            else:
                ffGridPair = calculateFFGridPair(mesh1, mesh2)

            for i in range(0, 121):
                for j in range(0, 121):
                    ffGrid[121*index1+i][121*index2+j] = ffGridPair[i][j]

    return ffGrid

def drawLine(vertex0X, vertex0Y, vertex1X, vertex1Y, colourStart, colourEnd, screen):
     xDiff = vertex1X - vertex0X
     yDiff = vertex1Y - vertex0Y
     numSteps = max(abs(xDiff), abs(yDiff))

     if (numSteps == 0):
         screen[round(vertex0Y)][round(vertex0X)] = colourStart
     else:
         xStep = xDiff/numSteps
         yStep = yDiff/numSteps

         colourDiff = colourEnd - colourStart
         colourStep = colourDiff/numSteps

         for step in range(0, numSteps+1):
             screen[round(vertex0Y + yStep*step)][round(vertex0X + xStep*step)] = colourStart + step*colourStep


def calculateLine(vertex0X, vertex0Y, vertex1X, vertex1Y):
    xDiff = vertex1X - vertex0X
    yDiff = vertex1Y - vertex0Y
    numSteps = max(abs(xDiff), abs(yDiff))

    xStep = xDiff/numSteps
    yStep = yDiff/numSteps

    lineList = []
    for step in range(0, numSteps+1):
        lineList.append((round(vertex0X + xStep*step), round(vertex0Y + yStep*step)))

    return lineList

def fillPatch(line0, line1, line2, line3, colour0, colour1, colour2, colour3, screen):
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

    colourYDiffLeft = colour0 - colour1
    colourYDiffRight = colour2 - colour3

    colourYLeftStep = colourYDiffLeft/(yDiff-1)
    colourYRightStep = colourYDiffRight/(yDiff-1)


    for index in range(0, yDiff):
        colourStart = colour3 + index*colourYRightStep
        colourEnd   = colour1 + index*colourYLeftStep

        drawLine(minXs[index], minY+index, maxXs[index], minY+index, colourStart, colourEnd, screen)

def projectVertex(vertex, focalLength, width, height):
    projectedVertex = vertex*focalLength/vertex[2] + torch.tensor([width/2, height/2, 0.0])

    #negating due to swap in direction of X and Y
    projectedVertexX = round(-1.0*projectedVertex[0].item())
    projectedVertexY = round(-1.0*projectedVertex[1].item())

    return projectedVertexX, projectedVertexY



def projectToScreen(meshList, colours, width, height):

    focalLength = height/2
    cameraPos = torch.tensor([278.0, 273.0, -600.0])

    screen = torch.zeros(height, width, 3)

    for meshIndex, mesh in enumerate(meshList):
        for vertIndex, vertex in enumerate(mesh.meshGrid):
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

            if (((vertIndex+1)%11 != 0) and vertIndex < 109):

                light0 = colours[121*meshIndex + vertIndex].item()
                light1 = colours[121*meshIndex + vertIndex + 1].item()
                light2 = colours[121*meshIndex + vertIndex + 11].item()
                light3 = colours[121*meshIndex + vertIndex + 12].item()

                colour0 = torch.tensor([255.0, 255.0, 255.0])*light0
                colour0 = torch.clamp(colour0, min=0.0, max=255.0)

                colour1 = torch.tensor([255.0, 255.0, 255.0])*light1
                colour1 = torch.clamp(colour1, min=0.0, max=255.0)

                colour2 = torch.tensor([255.0, 255.0, 255.0])*light2
                colour2 = torch.clamp(colour2, min=0.0, max=255.0)

                colour3 = torch.tensor([255.0, 255.0, 255.0])*light3
                colour3 = torch.clamp(colour3, min=0.0, max=255.0)

                #TODO: Fix colour indexing when forming cartesian product
                if ((meshIndex == 0) or (meshIndex == 2)):
                    fillPatch(line0, line1, line2, line3, colour0, colour1, colour2, colour3, screen)

                if (meshIndex == 1):
                    fillPatch(line0, line1, line2, line3, colour1, colour0, colour3, colour2, screen)

                if (meshIndex == 3):
                    fillPatch(line0, line1, line2, line3, colour0, colour2, colour1, colour3, screen)

                if (meshIndex == 4):
                    fillPatch(line0, line1, line2, line3, colour1, colour3, colour0, colour2, screen)
                #screen[projectedVertex1Y][projectedVertex1X] = torch.tensor([255.0, 255.0, 255.0])
                #screen[projectedVertex2Y][projectedVertex2X] = torch.tensor([255.0, 0.0, 0.0])
                #screen[projectedVertex3Y][projectedVertex3X] = torch.tensor([0.0, 0.0, 255.0])

    return screen

def writePPM(data, width, height, fileName):

    max = str(int(torch.max(data).item()))

    f = open(fileName, "w+")

    f.write("P3 \n")
    f.write(str(width) + " " + str(height) + "\n")
    f.write(max + " \n")

    for y in range(0, height):
        for x in range(0, width):
            f.write(str(int(data[y][x][0].item())) + " ")
            f.write(str(int(data[y][x][1].item())) + " ")
            f.write(str(int(data[y][x][2].item())) + " ")
            f.write(" ")
        f.write("\n")

    f.close()


class FFNet(nn.Module):
    def __init__(self, ffGrid):
        super(FFNet, self).__init__()
        self.fc = nn.Linear(numPatches, numPatches)
        self.fc.weight = torch.nn.Parameter(ffGrid)
        bias = torch.zeros_like(self.fc.bias)
        bias[171] = 15.0
        bias[172] = 15.0
        bias[183] = 15.0
        bias[184] = 15.0
        bias[195] = 15.0
        bias[196] = 15.0
        self.fc.bias = torch.nn.Parameter(bias)


    def forward(self, x):
        with torch.no_grad():
            for bounces in range(0, numBounces):
                x = self.fc(x)
        return x



numPatches = 605
numBounces = 3
pickleSave = False #TODO: make command line arg
# Create random Tensor to hold input
x = torch.zeros(1, numPatches)
#print(x)
#print(torch.sum(x))

floorMesh     = createBasicMesh(0.0, 556.0, 0.0, 0.0, 0.0, 559.2, 10)
ceilingMesh   = createBasicMesh(0.0, 556.0, 548.8, 548.8, 0.0, 559.2, 10)
backWallMesh  = createBasicMesh(0.0, 556.0, 0.0, 548.8, 559.2, 559.2, 10)
rightWallMesh = createBasicMesh(0.0, 0.0, 0.0, 548.8, 0.0, 559.2, 10)
leftWallMesh  = createBasicMesh(556.0, 556.0, 0.0, 548.8, 0.0, 559.2, 10)

meshList = [floorMesh, ceilingMesh, backWallMesh, rightWallMesh, leftWallMesh]

if (pickleSave):

    #print(time.perf_counter())
    ffGrid = calculateFFGrid(meshList) #TODO: Pickle this and load in values
    #print(time.perf_counter())
    #print(ffGrid)
    #print(torch.max(ffGrid))

    pickle_out = open("ffGridVertex.pickle","wb")
    pickle.dump(ffGrid, pickle_out)
    pickle_out.close()

else:

    pickle_in = open("ffGridVertex.pickle","rb")
    ffGrid = pickle.load(pickle_in)
    pickle_in.close()

model = FFNet(ffGrid)

start = time.perf_counter()
rad = model(x)
stop = time.perf_counter()
print("runtime: " + str(stop - start))
rad = torch.squeeze(rad)
#print(rad)
#print(rad.size())

width = 512
height = 512

data = projectToScreen(meshList, rad, width, height)

writePPM(data, width, height, "output.ppm")
