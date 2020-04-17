import torch
import torch.nn as nn

import math
import time
import pickle
import argparse

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
#{  130.0,  165.0,   65.0 }, top
#{   82.0,  165.0,  225.0 },
#{  242.0,  165.0,  274.0 },
#{  290.0,  165.0,  114.0 },

#{  290.0,    0.0,  114.0 }, left side
#{  290.0,  165.0,  114.0 },
#{  240.0,  165.0,  272.0 },
#{  240.0,    0.0,  272.0 },

#{  130.0,    0.0,   65.0 },
#{  130.0,  165.0,   65.0 },
#{  290.0,  165.0,  114.0 },
#{  290.0,    0.0,  114.0 }, front

#{   82.0,    0.0,  225.0 }, right side
#{   82.0,  165.0,  225.0 },
#{  130.0,  165.0,   65.0 },
#{  130.0,    0.0,   65.0 },

#{  240.0,    0.0,  272.0 }, back
#{  240.0,  165.0,  272.0 },
#{   82.0,  165.0,  225.0 },
#{   82.0,    0.0,  225.0 },

# Tall block -- white lambert
#{  423.0f,  330.0f,  247.0f, 0.0f }, top
#{  265.0f,  330.0f,  296.0f, 0.0f },
#{  314.0f,  330.0f,  455.0f, 0.0f },
#{  472.0f,  330.0f,  406.0f, 0.0f },

#{  423.0f,    0.0f,  247.0f, 0.0f },
#{  423.0f,  330.0f,  247.0f, 0.0f },
#{  472.0f,  330.0f,  406.0f, 0.0f }, left side
#{  472.0f,    0.0f,  406.0f, 0.0f },

#{  472.0f,    0.0f,  406.0f, 0.0f },
#{  472.0f,  330.0f,  406.0f, 0.0f },
#{  314.0f,  330.0f,  456.0f, 0.0f }, back
#{  314.0f,    0.0f,  456.0f, 0.0f },

#{  314.0f,    0.0f,  456.0f, 0.0f },
#{  314.0f,  330.0f,  456.0f, 0.0f }, right side
#{  265.0f,  330.0f,  296.0f, 0.0f },
#{  265.0f,    0.0f,  296.0f, 0.0f },

#{  265.0f,    0.0f,  296.0f, 0.0f }, front
#{  265.0f,  330.0f,  296.0f, 0.0f },
#{  423.0f,  330.0f,  247.0f, 0.0f },
#{  423.0f,    0.0f,  247.0f, 0.0f },


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
        self.area = self.calculateArea()
        self.norm = self.calculateNorm()

    def calculateArea(self):
        vec1Size = math.sqrt(self.vec1[0]*self.vec1[0] + self.vec1[1]*self.vec1[1] + self.vec1[2]*self.vec1[2])
        vec2Size = math.sqrt(self.vec2[0]*self.vec2[0] + self.vec2[1]*self.vec2[1] + self.vec2[2]*self.vec2[2])

        return vec1Size*vec2Size

    def calculateNorm(self):
        norm = torch.cross(self.vec1, self.vec2, dim=0)
        norm = torch.renorm(torch.unsqueeze(norm,0), p=2, dim=0, maxnorm=1)
        norm = torch.squeeze(norm)
        #centre = torch.tensor([278.0, 274.4, 279.6]) - self.meshGrid[0]
        centre = torch.tensor([278.0, 273.0, -600.0]) - self.meshGrid[0]
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


def calculateVisibility(start, rayDir, maxLength, index1, index2):
    #create box faces lists here
    cornerList = [torch.tensor([242.0, 165.0, 274.0]), torch.tensor([240.0, 165.0, 272.0]), torch.tensor([130.0, 165.0, 65.0]),
                  torch.tensor([290.0, 165.0, 114.0]), torch.tensor([472.0, 330.0, 406.0]), torch.tensor([472.0, 330.0, 406.0]),
                  torch.tensor([265.0, 330.0, 296.0]), torch.tensor([423.0, 330.0, 247.0])]

    v1List     = [torch.tensor([48.0, 0.0, -160.0]), torch.tensor([0.0, -165.0, 0.0]), torch.tensor([0.0, -165.0, 0.0]),
                  torch.tensor([0.0, -165.0, 0.0]), torch.tensor([-49.0, 0.0, -159.0]), torch.tensor([0.0, -330.0, 0.0]),
                  torch.tensor([0.0, -330.0, 0.0]), torch.tensor([0.0, -330.0, 0.0])]

    v2List     = [torch.tensor([-160.0, 0.0, -49.0]), torch.tensor([50.0, 0.0, -158.0]), torch.tensor([-48.0, 0.0, 160.0]),
                  torch.tensor([-160.0, 0.0, -49.0]), torch.tensor([-158.0, 0.0, 49.0]), torch.tensor([-49.0, 0.0, -159.0]),
                  torch.tensor([49.0, 0.0, 160.0]), torch.tensor([-158.0, 0.0, 49.0])]


    #remove indices
    smallerIndex = min(index1, index2)
    largerIndex = max(index1, index2)

    if (largerIndex > 4):

        del cornerList[largerIndex - 5]
        del v1List[largerIndex - 5]
        del v2List[largerIndex - 5]

        if (smallerIndex > 4):
            del cornerList[smallerIndex - 5]
            del v1List[smallerIndex - 5]
            del v2List[smallerIndex - 5]


    #check for intersection with blocks
    for index in range(0, len(cornerList)):
        A = torch.zeros(3,3)
        for i in range(0,3):
            A[i][0] = -1.0*rayDir[i].item()
            A[i][1] = v1List[index][i].item()
            A[i][2] = v2List[index][i].item()
        B = torch.unsqueeze(start - cornerList[index], 1)

        if (A.det() != 0):

            coeffs = torch.squeeze(torch.mm(A.inverse(), B))

            T = coeffs[0].item()
            U = coeffs[1].item()
            V = coeffs[2].item()
            if ((0 < T < maxLength-10) and (0 < U < 1) and (0 < V < 1)):
                return False

    return True


def calculateFFGridPair(mesh1, mesh2, meshIndex1, meshIndex2):
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

                if calculateVisibility(vertex2, diffDir, diffLength, meshIndex1, meshIndex2):

                    vertex1Norm = mesh1.norm
                    vertex2Norm = mesh2.norm
                    dot1 = -1.0*(vertex1Norm[0]*diffDir[0] + vertex1Norm[1]*diffDir[1] + vertex1Norm[2]*diffDir[2])
                    dot1 = max(dot1, 0.0)
                    dot2 = vertex2Norm[0]*diffDir[0] + vertex2Norm[1]*diffDir[1] + vertex2Norm[2]*diffDir[2]
                    dot2 = max(dot2, 0.0)
                    area2 = mesh2.area

                    ffGrid[index1][index2] = dot1*dot2*area2/(3.14*diffLength*diffLength)

                else:
                    ffGrid[index1][index2] = 0.0


    return ffGrid

def calculateFFGrid(meshList, numPatches):
    colourValuesR = [0.8, 0.05, 0.8]
    colourValuesG = [0.8, 0.8, 0.05]
    colourValuesB = [0.8, 0.05, 0.05]

    colourIndexList = [0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0]

    colours = torch.zeros(3*numPatches)
    ffGrid = torch.zeros(3*numPatches, 3*numPatches)
    for index1, mesh1 in enumerate(meshList):
        for index2, mesh2 in enumerate(meshList):
            mesh1Size = mesh1.meshGrid.size()[0]
            mesh2Size = mesh2.meshGrid.size()[0]
            if (index1 == index2):
                ffGridPair = torch.zeros(mesh1Size, mesh2Size)
            else:
                ffGridPair = calculateFFGridPair(mesh1, mesh2, index1, index2)

            for i in range(0, mesh1Size):
                for j in range(0, mesh2Size):
                    ffGrid[3*mesh1Size*index1+3*i][3*mesh2Size*index2+3*j] = ffGridPair[i][j]
                    ffGrid[3*mesh1Size*index1+3*i+1][3*mesh2Size*index2+3*j+1] = ffGridPair[i][j]
                    ffGrid[3*mesh1Size*index1+3*i+2][3*mesh2Size*index2+3*j+2] = ffGridPair[i][j]

                colours[3*mesh1Size*index1+3*i] = colourValuesR[colourIndexList[index1]]
                colours[3*mesh1Size*index1+3*i+1] = colourValuesG[colourIndexList[index1]]
                colours[3*mesh1Size*index1+3*i+2] = colourValuesB[colourIndexList[index1]]

    return ffGrid, colours

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

    lineList = []
    if (numSteps == 0):
        lineList.append(vertex0)

    else:
        xStep = xDiff/numSteps
        yStep = yDiff/numSteps
        colourStep = colourDiff/numSteps

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

                r0 = colours[3*meshSize*meshIndex + 3*vertIndex].item()
                r1 = colours[3*(meshSize*meshIndex + vertIndex + 1)].item()
                r2 = colours[3*(meshSize*meshIndex + vertIndex + meshWidth + 1)].item()
                r3 = colours[3*(meshSize*meshIndex + vertIndex + meshWidth)].item()

                g0 = colours[3*(meshSize*meshIndex + vertIndex)+1].item()
                g1 = colours[3*(meshSize*meshIndex + vertIndex + 1)+1].item()
                g2 = colours[3*(meshSize*meshIndex + vertIndex + meshWidth + 1)+1].item()
                g3 = colours[3*(meshSize*meshIndex + vertIndex + meshWidth)+1].item()

                b0 = colours[3*(meshSize*meshIndex + vertIndex)+2].item()
                b1 = colours[3*(meshSize*meshIndex + vertIndex + 1)+2].item()
                b2 = colours[3*(meshSize*meshIndex + vertIndex + meshWidth + 1)+2].item()
                b3 = colours[3*(meshSize*meshIndex + vertIndex + meshWidth)+2].item()

                colour0 = torch.tensor([r0, g0, b0])*255.0
                colour0 = torch.clamp(colour0, min=0.0, max=255.0)

                colour1 = torch.tensor([r1, g1, b1])*255.0
                colour1 = torch.clamp(colour1, min=0.0, max=255.0)

                colour2 = torch.tensor([r2, g2, b2])*255.0
                colour2 = torch.clamp(colour2, min=0.0, max=255.0)

                colour3 = torch.tensor([r3, g3, b3])*255.0
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
    def __init__(self, ffGrid, colours):
        super(FFNet, self).__init__()
        self.colours = colours
        self.fc = nn.Linear(3*numPatches, 3*numPatches)
        self.fc.weight = torch.nn.Parameter(ffGrid)
        wallSize = int(numPatches/13)
        wallWidth = int(math.sqrt(wallSize))
        lightCorner = int(math.floor(wallSize + wallSize/2 - wallWidth))
        bias = torch.zeros_like(self.fc.bias)
        #print(lightCorner)
        #print(wallWidth)
        bias[3*(lightCorner-1)] = 15.0
        bias[3*(lightCorner)] = 15.0
        bias[3*(lightCorner+1)] = 15.0
        bias[3*(lightCorner+wallWidth-1)] = 15.0
        bias[3*(lightCorner+wallWidth)] = 15.0
        bias[3*(lightCorner+wallWidth+1)] = 15.0
        bias[3*(lightCorner-1)+1] = 15.0
        bias[3*(lightCorner)+1] = 15.0
        bias[3*(lightCorner+1)+1] = 15.0
        bias[3*(lightCorner+wallWidth-1)+1] = 15.0
        bias[3*(lightCorner+wallWidth)+1] = 15.0
        bias[3*(lightCorner+wallWidth+1)+1] = 15.0
        bias[3*(lightCorner-1)+2] = 15.0
        bias[3*(lightCorner)+2] = 15.0
        bias[3*(lightCorner+1)+2] = 15.0
        bias[3*(lightCorner+wallWidth-1)+2] = 15.0
        bias[3*(lightCorner+wallWidth)+2] = 15.0
        bias[3*(lightCorner+wallWidth+1)+2] = 15.0
        self.fc.bias = torch.nn.Parameter(bias)


    def forward(self, x):
        with torch.no_grad():
            for bounces in range(0, numBounces):
                x = self.fc(x)
                x = self.colours*x

        return x


numDivides = 10
numPatches = 13*(numDivides+1)*(numDivides+1)
numBounces = 3

parser = argparse.ArgumentParser(description='Parse command line arguments')

parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--pickle-save', action='store_true',
                    help='Calculate and Store Form Factors')

args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

with torch.no_grad():

    # Create Tensor to hold input
    x = torch.zeros(1, 3*numPatches).to(device=args.device)
    #print(x)
    #print(torch.sum(x))

    floorMesh     = createBasicMesh(torch.tensor([556.0, 0.0, 559.2]), torch.tensor([0.0, 0.0, -559.2]), torch.tensor([-556.0, 0.0, 0.0]), numDivides)
    ceilingMesh   = createBasicMesh(torch.tensor([556.0, 548.8, 559.2]), torch.tensor([0.0, 0.0, -559.2]), torch.tensor([-556.0, 0.0, 0.0]), numDivides)
    backWallMesh  = createBasicMesh(torch.tensor([556.0, 548.8, 559.2]), torch.tensor([0.0, -548.8, 0.0]), torch.tensor([-556.0, 0.0, 0.0]), numDivides)
    rightWallMesh = createBasicMesh(torch.tensor([0.0, 548.8, 559.2]), torch.tensor([0.0, -548.8, 0.0]), torch.tensor([0.0, 0.0, -559.2]), numDivides)
    leftWallMesh  = createBasicMesh(torch.tensor([556.0, 548.8, 0.0]), torch.tensor([0.0, -548.8, 0.0]), torch.tensor([0.0, 0.0, 559.2]), numDivides)

    shortBlockTop   = createBasicMesh(torch.tensor([242.0, 165.0, 274.0]), torch.tensor([48.0, 0.0, -160.0]), torch.tensor([-160.0, 0.0, -49.0]), numDivides)
    shortBlockLeft  = createBasicMesh(torch.tensor([240.0, 165.0, 272.0]), torch.tensor([0.0, -165.0, 0.0]), torch.tensor([50.0, 0.0, -158.0]), numDivides)
    shortBlockRight = createBasicMesh(torch.tensor([130.0, 165.0, 65.0]), torch.tensor([0.0, -165.0, 0.0]), torch.tensor([-48.0, 0.0, 160.0]), numDivides)
    shortBlockFront = createBasicMesh(torch.tensor([290.0, 165.0, 114.0]), torch.tensor([0.0, -165.0, 0.0]), torch.tensor([-160.0, 0.0, -49.0]), numDivides)
    #shortBlockBack  = createBasicMesh(torch.tensor([82.0, 165.0, 225.0]), torch.tensor([0.0, -165.0, 0.0]), torch.tensor([158.0, 0.0, 47.0]), numDivides)

    tallBlockTop   = createBasicMesh(torch.tensor([472.0, 330.0, 406.0]), torch.tensor([-49.0, 0.0, -159.0]), torch.tensor([-158.0, 0.0, 49.0]), numDivides)
    tallBlockLeft  = createBasicMesh(torch.tensor([472.0, 330.0, 406.0]), torch.tensor([0.0, -330.0, 0.0]), torch.tensor([-49.0, 0.0, -159.0]), numDivides)
    tallBlockRight = createBasicMesh(torch.tensor([265.0, 330.0, 296.0]), torch.tensor([0.0, -330.0, 0.0]), torch.tensor([49.0, 0.0, 160.0]), numDivides)
    tallBlockFront = createBasicMesh(torch.tensor([423.0, 330.0, 247.0]), torch.tensor([0.0, -330.0, 0.0]), torch.tensor([-158.0, 0.0, 49.0]), numDivides)
    #tallBlockBack  = createBasicMesh(torch.tensor([314.0, 330.0, 456.0]), torch.tensor([0.0, -330.0, 0.0]), torch.tensor([158.0, 0.0, -50.0]), numDivides)

    meshList = [floorMesh, ceilingMesh, backWallMesh, rightWallMesh, leftWallMesh, tallBlockLeft, tallBlockRight, tallBlockFront, tallBlockTop, shortBlockLeft, shortBlockRight, shortBlockFront, shortBlockTop]

    if (args.pickle_save):

        #print(time.perf_counter())
        ffGrid, colours = calculateFFGrid(meshList, numPatches)
        #print(time.perf_counter())
        #print(ffGrid)
        #print(torch.max(ffGrid))

        pickle_out = open("ffGridVertex10BoxesColourWide.pickle","wb")
        pickle.dump((ffGrid, colours), pickle_out)
        pickle_out.close()

    else:

        pickle_in = open("ffGridVertex10BoxesColourWide.pickle","rb")
        ffGrid, colours = pickle.load(pickle_in)
        pickle_in.close()

    ffGrid = ffGrid.to(device=args.device)
    model = FFNet(ffGrid, colours).to(device=args.device)

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

    writePPM(data, width, height, "outputC.ppm")
