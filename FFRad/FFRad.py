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

#Short block -- white lambert
#{  130.0,  165.0,   65.0 },
#{   82.0,  165.0,  225.0 },
#{  242.0,  165.0,  274.0 },
#{  290.0,  165.0,  114.0 },

#{  290.0,    0.0,  114.0 },
#{  290.0,  165.0,  114.0 },
#{  240.0,  165.0,  272.0 },
#{  240.0,    0.0,  272.0 },

#{  130.0,    0.0,   65.0 },
#{  130.0,  165.0,   65.0 },
#{  290.0,  165.0,  114.0 },
#{  290.0,    0.0,  114.0 },

#{   82.0,    0.0,  225.0 },
#{   82.0,  165.0,  225.0 },
#{  130.0,  165.0,   65.0 },
#{  130.0,    0.0,   65.0 },

#{  240.0,    0.0,  272.0 },
#{  240.0,  165.0,  272.0 },
#{   82.0,  165.0,  225.0 },
#{   82.0,    0.0,  225.0 },


class Vertex:
    def __init__(self, x, y, colour):
        self.x = x
        self.y = y
        self.colour = colour


class Mesh:
    def __init__(self, vec1, vec2, meshGrid, centreGrid):
        self.vec1 = vec1
        self.vec2 = vec2
        self.meshGrid = meshGrid
        self.centreGrid = centreGrid

    def area(self):
        vec1Size = math.sqrt(self.vec1[0]*self.vec1[0] + self.vec1[1]*self.vec1[1] + self.vec1[2]*self.vec1[2])
        vec2Size = math.sqrt(self.vec2[0]*self.vec2[0] + self.vec2[1]*self.vec2[1] + self.vec2[2]*self.vec2[2])

        return vec1Size*vec2Size

    def norm(self, vertex):
        norm = torch.cross(self.vec1, self.vec2, dim=0)
        norm = torch.renorm(torch.unsqueeze(norm,0), p=2, dim=0, maxnorm=1)
        norm = torch.squeeze(norm)
        centre = torch.tensor([278.0, 274.4, 279.6]) - vertex
        centreDot = norm[0]*centre[0] + norm[1]*centre[1] + norm[2]*centre[2]
        if (centreDot < 0):
            norm = -1.0*norm

        return norm

    def calculatePatchVertices(self, vertex):
        vertex1 = vertex + self.vec2
        vertex2 = vertex1 + self.vec1
        vertex3 = vertex2 - self.vec2

        return vertex1, vertex2, vertex3


def createBasicMesh(corner, fullVec1, fullVec2, numDivides):
    vec1 = fullVec1/numDivides
    vec2 = fullVec2/numDivides

    meshGrid = torch.zeros((numDivides+1)*(numDivides+1), 3)
    for i in range(0, numDivides+1):
        for j in range(0, numDivides+1):
            meshGrid[i*(numDivides+1) + j] = corner + i*vec1 + j*vec2


    centreGrid = torch.add(meshGrid, vec1/2.0 + vec2/2.0)
    #print(centreGrid.size())

    mesh = Mesh(vec1, vec2, meshGrid, centreGrid)

    return mesh


def calculateFFGridPair(mesh1, mesh2):
    mesh1Size = mesh1.meshGrid.size()[0]
    mesh2Size = mesh2.meshGrid.size()[0]
    ffGrid = torch.zeros(mesh1Size, mesh2Size)

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

def calculateFFGrid(meshList, numPatches):
    ffGrid = torch.zeros(numPatches, numPatches)
    for index1, mesh1 in enumerate(meshList):
        for index2, mesh2 in enumerate(meshList):
            mesh1Size = mesh1.meshGrid.size()[0]
            mesh2Size = mesh2.meshGrid.size()[0]
            if (index1 == index2):
                ffGridPair = torch.zeros(mesh1Size, mesh2Size)
            else:
                ffGridPair = calculateFFGridPair(mesh1, mesh2)

            for i in range(0, mesh1Size):
                for j in range(0, mesh2Size):
                    ffGrid[mesh1Size*index1+i][mesh2Size*index2+j] = ffGridPair[i][j]

    return ffGrid

def drawLine(vertex0, vertex1, screen):
     xDiff = vertex1.x - vertex0.x
     yDiff = vertex1.y - vertex0.y
     numSteps = max(abs(xDiff), abs(yDiff))

     if (numSteps == 0):
         screen[round(vertex0.y)][round(vertex0.x)] = vertex0.colour
     else:
         xStep = xDiff/numSteps
         yStep = yDiff/numSteps

         colourDiff = vertex1.colour - vertex0.colour
         colourStep = colourDiff/numSteps

         for step in range(0, numSteps+1):
             screen[round(vertex0.y + yStep*step)][round(vertex0.x + xStep*step)] = vertex0.colour + step*colourStep


def calculateLine(vertex0, vertex1):
    xDiff = vertex1.x - vertex0.x
    yDiff = vertex1.y - vertex0.y
    colourDiff = vertex1.colour - vertex0.colour
    numSteps = max(abs(xDiff), abs(yDiff))

    xStep = xDiff/numSteps
    yStep = yDiff/numSteps
    colourStep = colourDiff/numSteps

    lineList = []
    for step in range(0, numSteps+1):
        newVertex = Vertex(round(vertex0.x + xStep*step), round(vertex0.y + yStep*step), vertex0.colour + colourStep*step)
        lineList.append(newVertex)

    return lineList

def fillPatch(line0, line1, line2, line3, screen):
    maxY = int(max(line0[0].y, line1[0].y, line2[0].y, line3[0].y))
    minY = int(min(line0[0].y, line1[0].y, line2[0].y, line3[0].y))
    yDiff = (maxY - minY)+1

    maxXs = [Vertex(-math.inf, 0, torch.tensor([0.0, 0.0, 0.0]))]*yDiff
    minXs = [Vertex(math.inf, 0, torch.tensor([0.0, 0.0, 0.0]))]*yDiff

    for pixel in line0:
        if (pixel.x < minXs[pixel.y-minY].x):
            minXs[pixel.y-minY] = pixel
        if (pixel.x > maxXs[pixel.y-minY].x):
            maxXs[pixel.y-minY] = pixel

    for pixel in line1:
        if (pixel.x < minXs[pixel.y-minY].x):
            minXs[pixel.y-minY] = pixel
        if (pixel.x > maxXs[pixel.y-minY].x):
            maxXs[pixel.y-minY] = pixel

    for pixel in line2:
        if (pixel.x < minXs[pixel.y-minY].x):
            minXs[pixel.y-minY] = pixel
        if (pixel.x > maxXs[pixel.y-minY].x):
            maxXs[pixel.y-minY] = pixel

    for pixel in line3:
        if (pixel.x < minXs[pixel.y-minY].x):
            minXs[pixel.y-minY] = pixel
        if (pixel.x > maxXs[pixel.y-minY].x):
            maxXs[pixel.y-minY] = pixel

    for index in range(0, yDiff):
        drawLine(minXs[index], maxXs[index], screen)


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
            meshSize = int(mesh.meshGrid.size()[0])
            meshWidth = int(math.sqrt(meshSize)) #only works if width and height divided equally
            vertex0Pos = vertex - cameraPos
            vertex1Pos, vertex2Pos, vertex3Pos = mesh.calculatePatchVertices(vertex0Pos)

            #if (((vertIndex+1)%11 != 0) and vertIndex < 109):
            if (((vertIndex+1)%meshWidth != 0) and vertIndex < (meshSize-meshWidth)):

                light0 = colours[meshSize*meshIndex + vertIndex].item()
                light1 = colours[meshSize*meshIndex + vertIndex + 1].item()
                light2 = colours[meshSize*meshIndex + vertIndex + meshWidth + 1].item()
                light3 = colours[meshSize*meshIndex + vertIndex + meshWidth].item()

                colour0 = torch.tensor([255.0, 255.0, 255.0])*light0
                colour0 = torch.clamp(colour0, min=0.0, max=255.0)

                colour1 = torch.tensor([255.0, 255.0, 255.0])*light1
                colour1 = torch.clamp(colour1, min=0.0, max=255.0)

                colour2 = torch.tensor([255.0, 255.0, 255.0])*light2
                colour2 = torch.clamp(colour2, min=0.0, max=255.0)

                colour3 = torch.tensor([255.0, 255.0, 255.0])*light3
                colour3 = torch.clamp(colour3, min=0.0, max=255.0)

                projectedVertex0X, projectedVertex0Y = projectVertex(vertex0Pos, focalLength, width, height)
                projectedVertex1X, projectedVertex1Y = projectVertex(vertex1Pos, focalLength, width, height)
                projectedVertex2X, projectedVertex2Y = projectVertex(vertex2Pos, focalLength, width, height)
                projectedVertex3X, projectedVertex3Y = projectVertex(vertex3Pos, focalLength, width, height)

                vertex0 = Vertex(projectedVertex0X, projectedVertex0Y, colour0)
                vertex1 = Vertex(projectedVertex1X, projectedVertex1Y, colour1)
                vertex2 = Vertex(projectedVertex2X, projectedVertex2Y, colour2)
                vertex3 = Vertex(projectedVertex3X, projectedVertex3Y, colour3)

                line0 = calculateLine(vertex0, vertex1)
                line1 = calculateLine(vertex1, vertex2)
                line2 = calculateLine(vertex2, vertex3)
                line3 = calculateLine(vertex3, vertex0)

                fillPatch(line0, line1, line2, line3, screen)

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
        wallSize = int(numPatches/5)
        wallWidth = int(math.sqrt(wallSize))
        lightCorner = int(math.floor(wallSize + wallSize/2 - wallWidth))
        bias = torch.zeros_like(self.fc.bias)
        #print(lightCorner)
        #print(wallWidth)
        bias[lightCorner-1] = 15.0
        bias[lightCorner] = 15.0
        bias[lightCorner+1] = 15.0
        bias[lightCorner+wallWidth-1] = 15.0
        bias[lightCorner+wallWidth] = 15.0
        bias[lightCorner+wallWidth+1] = 15.0
        self.fc.bias = torch.nn.Parameter(bias)


    def forward(self, x):
        with torch.no_grad():
            for bounces in range(0, numBounces):
                x = self.fc(x)
        return x


numDivides = 10
numPatches = 5*(numDivides+1)*(numDivides+1)
numBounces = 3
pickleSave = False #TODO: make command line arg
# Create Tensor to hold input
x = torch.zeros(1, numPatches)
#print(x)
#print(torch.sum(x))

floorMesh     = createBasicMesh(torch.tensor([556.0, 0.0, 559.2]), torch.tensor([0.0, 0.0, -559.2]), torch.tensor([-556.0, 0.0, 0.0]), numDivides)
ceilingMesh   = createBasicMesh(torch.tensor([556.0, 548.8, 559.2]), torch.tensor([0.0, 0.0, -559.2]), torch.tensor([-556.0, 0.0, 0.0]), numDivides)
backWallMesh  = createBasicMesh(torch.tensor([556.0, 548.8, 559.2]), torch.tensor([0.0, -548.8, 0.0]), torch.tensor([-556.0, 0.0, 0.0]), numDivides)
rightWallMesh = createBasicMesh(torch.tensor([0.0, 548.8, 559.2]), torch.tensor([0.0, -548.8, 0.0]), torch.tensor([0.0, 0.0, -559.2]), numDivides)
leftWallMesh  = createBasicMesh(torch.tensor([556.0, 548.8, 0.0]), torch.tensor([0.0, -548.8, 0.0]), torch.tensor([0.0, 0.0, 559.2]), numDivides)

meshList = [floorMesh, ceilingMesh, backWallMesh, rightWallMesh, leftWallMesh]

if (pickleSave):

    #print(time.perf_counter())
    ffGrid = calculateFFGrid(meshList, numPatches)
    #print(time.perf_counter())
    #print(ffGrid)
    #print(torch.max(ffGrid))

    pickle_out = open("ffGridVertex10Boxes.pickle","wb")
    pickle.dump(ffGrid, pickle_out)
    pickle_out.close()

else:

    pickle_in = open("ffGridVertex10Boxes.pickle","rb")
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
