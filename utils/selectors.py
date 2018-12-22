import os
import re

def selectParam(paramsPath):
    paramFiles = [] #get all param files
    epochs = [] #get all epochs
    for fileName in os.listdir(paramsPath):
        if os.path.splitext(fileName)[1] == '.params':
            paramFiles.append(fileName.split('-')[0])
            epochs.append(int(re.search('-(\d*)\.', fileName).group(1)))

    for i, paramFile in enumerate(paramFiles):
        print('{}: {}-{}'.format(i, paramFile, epochs[i]))

    inputParamNo = input('input param file No.:')    #get selected paramFile and epoch
    param = paramFiles[inputParamNo]
    epoch = epochs[inputParamNo]

    return param, epoch

def selectImg(imgsPath):
    imgFiles = os.listdir(imgsPath)
    for i, imgFile in enumerate(imgFiles):
        print('{}: {}'.format(i, imgFile))

    inputImgNo = input('input image file No.:')  # get selected img file
    imgName = imgFiles[inputImgNo]

    return imgName

def selectLog(logsPath):
    logFiles = os.listdir(logsPath)
    for i, logFile in enumerate(logFiles):
        print('{}: {}'.format(i, logFile))

    logFileName = logFiles[input('input log file No.:')]

    return logFileName