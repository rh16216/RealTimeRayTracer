import torch

def parsePPM(fileName):
    f = open(fileName, "r")
    fileContents = f.read()
    fileLines = fileContents.split("\n")

    size = fileLines[1].split(" ")
    width = int(size[0])
    height = int(size[1])
    del fileLines[0:3]

    imageData = torch.zeros(width, height, 3)

    for lineIndex, line in enumerate(fileLines):
        lineValues = line.split("  ")
        for valueIndex, value in enumerate(lineValues):
            rgb = value.split(" ")
            if (len(rgb) == 3):
                imageData[lineIndex][valueIndex][0] = int(rgb[0])
                imageData[lineIndex][valueIndex][1] = int(rgb[1])
                imageData[lineIndex][valueIndex][2] = int(rgb[2])

    return imageData


def compareImages(image1, image2):
    if image1.size() == image2.size():
        width = image1.size()[0]
        height = image1.size()[1]

        diffData = torch.zeros(width, height, 3)
        for i in range(0, height):
            for j in range(0, width):
                diffData[i][j] = torch.abs(image1[i][j] - image2[i][j])

    return diffData


def writePPM(data, fileName):
    width = data.size()[0]
    height = data.size()[1]

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



image1 = parsePPM("10patches.ppm")
image2 = parsePPM("10blocks.ppm")

diff = compareImages(image1, image2)

writePPM(diff, "diff.ppm")
