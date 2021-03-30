import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from shutil import copyfile,rmtree

import sklearn
from sklearn import preprocessing

row2IndexMap = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
excel_file = "./data/HetY1H Pilot TF coordinates.xlsx"
df = pd.read_excel(excel_file, engine='openpyxl')  # , index_col=None, header=None)
css = '''
h1{
  font-size: 30px;
  color: #212121;
  text-transform: uppercase;
  font-weight: bolder;
  text-align: center;
  margin-bottom: 15px;
}
table{
  width:100%;
  table-layout: fixed;
  border-spacing:5px;
}

th{
  padding: 20px 15px;
  text-align: left;
  font-weight: bolder;
  font-size: 14px;
  color: #212121;
  text-transform: uppercase;
  text-align-last: center;
  box-shadow: 3px 3px 5px 6px #ccc;
}
td{
  padding: 15px;
  text-align: left;
  vertical-align:middle;
  font-weight: bolder;
  font-size: 14px;
  color: #212121;
  border-bottom: solid 1px rgba(0,0,0,0.1);
  box-shadow: 3px 3px 5px 6px #adabab;
}


/* demo styles */
body{
  font-family: 'Roboto', sans-serif;
  font-weight:bolder;
}
section{
  margin: 50px;
}

/* for custom scrollbar for webkit browser*/

::-webkit-scrollbar {
    width: 6px;
} 
::-webkit-scrollbar-track {
    -webkit-box-shadow: inset 0 0 6px rgba(0,0,0,0.3); 
} 
::-webkit-scrollbar-thumb {
    -webkit-box-shadow: inset 0 0 6px rgba(0,0,0,0.3); 
}'''
html = '''<html><head><title>{0}</title><style>{1}</style></head><body><div>{2}<div><body></html'''


def getIndex(inp, rows, cols):
    index = inp.split("-")

    yIndex = row2IndexMap[index[1][0]]
    xIndex = int(index[1][1:]) - 1
    if index[0] == "1":
        xIndex = 4 * xIndex
        yIndex = 4 * yIndex
    elif index[0] == "2":
        xIndex = 4 * xIndex + 2
        yIndex = 4 * yIndex
    elif index[0] == "3":
        xIndex = 4 * xIndex
        yIndex = 4 * yIndex + 2
    elif index[0] == "4":
        xIndex = 4 * xIndex + 2
        yIndex = 4 * yIndex + 2
    colonies = []
    for i in range(2):
        for j in range(2):
            colonies.append([[rows[xIndex + i] - rows[xIndex], rows[xIndex + 1 + i] - rows[xIndex]],
                             [cols[yIndex + j] - cols[yIndex], cols[yIndex + j + 1] - cols[yIndex]]])
    return [[rows[xIndex], rows[xIndex + 2]], [cols[yIndex], cols[yIndex + 2]]], colonies


import glob


def rescaleIntensity(img, range):
    table = np.interp(np.arange(256), range, [0, 255]).astype('uint8')
    return cv2.LUT(img, table)


def getHistogramProjection(img):
    histogram = img.copy()
    histogram[histogram == 255] = 1
    return np.sum(histogram, axis=0), np.sum(histogram, axis=1)


def imageTageGenerator(path):
    return '''<img src="{}">'''.format(path)


def genPercentage(val):
    return str(val) + "%"


def getTruth(val):
    global activityThreshold
    if val < activityThreshold:
        return '<b style="color:red;">' + str(round(val, 3)) + '</b>'
    else:
        return '<b style="color:green;">' + str(round(val, 3)) + '</b>'


def getPeaks(proj, thresh1, thresh2):
    gradient = np.gradient(proj)
    gradient = (np.logical_or(gradient > thresh1, gradient < thresh2)) * gradient
    peaks = []
    isPeak = False
    lastPeak = 0
    for i in range(1, len(gradient)):
        if gradient[i] > 0:
            isPeak = True
        if gradient[i] < 0 and isPeak and (i - lastPeak) > 70 and proj[i] > 100:
            peaks.append(i)
            lastPeak = i
            isPeak = False
    diff = np.diff(peaks)
    avg = int(np.mean(diff))
    for i, v in enumerate(diff):
        if math.fabs(v - 2 * avg) < 5:
            peaks.insert(i + 1, peaks[i] + avg)
    peaks.insert(len(peaks), peaks[-1] + avg)
    peaks = np.subtract(peaks, int(avg / 2))
    return peaks, gradient


def getLevelRange(img, percentile=0.01):
    # Assuming grayscale
    histogram = np.histogram(img.flatten(), range(257))
    cumulativeSum = np.cumsum(histogram[0])
    lower = upper = 255
    for i, j in enumerate(cumulativeSum):
        if i < lower and j >= percentile * cumulativeSum[-1]:
            lower = i
        if upper > i and j >= (1 - percentile) * cumulativeSum[-1]:
            upper = i
            break
    # print("total=",cumulativeSum[-1],"lower:",cumulativeSum[lower-2:lower+2],"upper:",cumulativeSum[upper-2:upper+2])
    # print(cumulativeSum[lower],cumulativeSum[upper])
    return [lower, upper]


def createProjectionImages(gray, xProj, yProj, path='./'):
    height, width = gray.shape
    blankImage = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        cv2.line(blankImage, (0, row), (int(yProj[row]), row), (255, 255, 255), 1)
    cv2.imwrite(path + 'Horz.jpg', blankImage)
    blankImage = np.zeros((height, width, 3), dtype=np.uint8)
    for column in range(width):
        cv2.line(blankImage, (column, 0), (column, int(xProj[column])), (255, 255, 255), 1)
    cv2.imwrite(path + 'vert.jpg', blankImage)

activityThreshold = 0.01

if len(sys.argv) == 1:
    print("Usage test.py <path_to_images>")
    exit(-1)
p = sys.argv[1]
p = p[:-1] if p[-1] == "/" else p
path = glob.glob(sys.argv[1] + "/*.JPG")
dataframes=[]
for imagePath in path:
    name = Path(imagePath).stem
    name = name.split('.')
    outputPath = "./output/{}".format(name[0])
    clusterPath = "./output/cluster/"
    rmtree(clusterPath,ignore_errors=True)
    for i in range(10, 110, 10):
        Path(clusterPath + str(i)).mkdir(parents=True, exist_ok=True)
    dataframe = pd.read_excel(name[0] + ".xlsx", engine='openpyxl')
    image = cv2.imread(imagePath)
    if image is None:
        print("Can't access image at ", sys.argv[1])
        exit(-1)
    img = cv2.Canny(image, 20, 30)
    cv2.imwrite('edge-detect.jpg', img)
    template1 = cv2.imread('upperLeft.png', 0)
    template2 = cv2.imread('bottomRight.png', 0)

    res = cv2.matchTemplate(img, template1, cv2.TM_CCORR_NORMED)
    res1 = cv2.matchTemplate(img, template2, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = np.add(max_loc, [140, 40])
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res1)
    bottom_right = np.add(max_loc, [0, 140])
    img = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    cv2.imwrite('first crop.png', img)
    # cv2.waitKey()
    # cv2.rectangle(img,tuple(top_left), tuple(bottom_right), 255, 2)
    # cv2.imwrite('crop.jpg',img1[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]])
    # cv2.waitKey(0)
    # print(top_left,bottom_right)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.Sobel(gray, cv2.CV_8UC1, 1, 0, ksize=5)
    gray = cv2.blur(gray, (5, 5))
    # hist,bins = np.histogram(img.ravel(),256,[0,256])
    gray = ((gray > 90) * 255).astype(np.uint8)
    cv2.imwrite('gray.jpg', gray)
    # cv2.waitKey()
    xProj, yProj = getHistogramProjection(gray)
    smooth = 5
    xProj = np.convolve(xProj, [1 / smooth for i in range(smooth)])[:-(smooth - 1)]
    yProj = np.convolve(yProj, [1 / smooth for i in range(smooth)])[:-(smooth - 1)]
    xProj = (xProj > 20) * xProj
    yProj = (yProj > 20) * yProj
    # for i in range(len(xProj)-2):
    #     xProj[i]= (xProj[i]+xProj[i+1]+xProj[i+2])/3

    xNonZero = [i for i, elem in enumerate(xProj) if elem > 30]
    yNonZero = [i for i, elem in enumerate(yProj) if elem > 30]
    # crop = img[yNonZero[0]:yNonZero[-1],xNonZero[0]:xNonZero[-1]]
    crop = img
    xpeaks, xGrad = getPeaks(xProj, 0, 0)

    ypeaks, yGrad = getPeaks(yProj, 0, 0)
    # print(len(xpeaks),len(ypeaks))
    outputPercent = []
    outputIntensity = []
    hsvCrop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # levelRange=getLevelRange(crop[:, :, 2])
    # rgbImage = rescaleIntensity(crop[:, :, 2], levelRange)
    # rgbImage =cv2.bitwise_not(rgbImage)
    # np.mean(hist2[0])
    # np.mean(hist3[0])
    # plt.figure(131)
    # plt.plot(range(256),t1)
    # plt.figure(132)
    # plt.hist(crop[:,:,2].ravel(),256,[0,256],density=True)
    # plt.figure(133)
    # plt.hist(gray.ravel(),256,[0,256])
    # plt.show()
    # cv2.imwrite(name[0]+"_hsv.png",hsvImage)
    # cv2.imwrite(name[0]+"_r.png",((crop[:,:,2]<thresh1)*crop[:,:,2]))
    # cv2.imwrite(name[0]+"_hsv.png",hsvImage)
    # cv2.imwrite(name[0]+"_r.png",rgbImage)
    outputImage = []
    Path(outputPath).mkdir(parents=True, exist_ok=True)
    reference, refCols = getIndex("3-B7", xpeaks, ypeaks)

    for c in df["Coordinate"]:
        if (len(c.split("-")[1]) > 1):
            index, colonies = getIndex(c, xpeaks, ypeaks)
            roi = (hsvCrop[index[1][0]:index[1][1], index[0][0]:index[0][1]])[:, :, 2]
            hsvImage = np.zeros_like(roi, dtype=np.uint8)
            for ix, colony in enumerate(colonies):
                subRoi = roi[colony[1][0]:colony[1][1], colony[0][0]:colony[0][1]]
                levelRange = getLevelRange(subRoi)
                # print(levelRange[1]-levelRange[0])
                if True:
                    colonyProcessed = rescaleIntensity(subRoi, [0,levelRange[1]])
                    # cv2.imshow('test', colonyProcessed)
                    # cv2.waitKey()
                    # colonyProcessed = cv2.medianBlur(colonyProcessed, 5)
                    colonyProcessed = cv2.GaussianBlur(colonyProcessed, (3, 3), 1.5)
                    # cv2.imshow('test', colonyProcessed)
                    # cv2.waitKey()
                    test = getLevelRange(colonyProcessed,0.05)
                    # if(c=="1-D6"):
                    #     print(name[0],test)
                    if test[1]-test[0]>=127:
                        test[0]=127
                    # test[0]=np.mean(test)
                    colonyProcessed = cv2.threshold(colonyProcessed, test[0], test[1], cv2.THRESH_BINARY_INV)[1]
                    # plt.imshow(colonyProcessed)
                    # plt.pause(0.01)
                    # colonyProcessed = cv2.erode(colonyProcessed,np.ones((3,3),np.uint8),iterations=1)
                    contours, hierarchy = cv2.findContours(colonyProcessed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    center=np.divide(colonyProcessed.shape,2)
                    centerContour=None
                    minDistance=colonyProcessed.size
                    for contour in contours:
                        moments = cv2.moments(contour)
                        if moments["m00"]!=0:
                            centroid= [int(moments["m10"] / moments["m00"]),int(moments["m01"] / moments["m00"])]
                            distance=math.sqrt(np.sum(np.square(centroid-center)))
                            if distance<minDistance:
                                centerContour=contour
                                minDistance=distance
                        # p=cv2.arcLength(contour,True)
                        # a=cv2.contourArea(contour)
                        # if p==0 or a==0 :
                        #     continue
                        # circularity= (4*math.pi*a)/p**2
                        # if math.fabs(1-circularity)<0.4 and maxArea<a:
                        #     circle=contour
                        #     maxArea=a
                    colonyProcessed = np.zeros_like(colonyProcessed,dtype=np.uint8)
                    # plt.show()

                    colonyProcessed=cv2.drawContours(colonyProcessed,[centerContour],0,(255,255,255),cv2.FILLED)
                    hsvImage[colony[1][0]:colony[1][1], colony[0][0]:colony[0][1]] = colonyProcessed
                    # if levelRange[1]-levelRange[0]<18:
                    #     print(levelRange[1] - levelRange[0], levelRange)
                    #     print(name[0],c)
                    # cv2.imshow('test',colonyProcessed)
                    # cv2.imshow('test2',hsvImage)
                    # cv2.waitKey()
                    # cv2.imshow('test3',roi)
                    # cv2.imshow('test4',crop[index[1][0]:index[1][1], index[0][0]:index[0][1]])

            # inv = cv2.bitwise_not(roi)
            inv =(hsvCrop[index[1][0]:index[1][1], index[0][0]:index[0][1]])[:, :, 0]
            inv = rescaleIntensity(inv,[getLevelRange(inv)[0],255])
            invRange = getLevelRange(inv,0)
            if c == "1-B9" or c == "1-B8":
                print(c,invRange)
            if invRange[1]-invRange[0]<3:
                inv = np.zeros_like(inv,dtype=np.uint8)
            roi = cv2.bitwise_and(inv, inv, mask=hsvImage)
            areaImage = roi.copy()
            areaImage[areaImage>0] = 1
            area = np.sum(areaImage)
            intensity = roi.copy()
            # hsvImage = cv2.bitwise_not(hsvImage)
            # roi = cv2.bitwise_and(roi,roi,mask=hsvImage)
            intensity = np.divide(intensity,np.subtract(256,intensity))
            outputIntensity.append(np.sum(intensity)*area)
            # outputPercent.append(np.mean(roi) - reference)
            # if c=="2-F7":
            #     cv2.imshow('test1',hsvImage)
            #     cv2.imshow('output',roi)
            #     cv2.imshow('inv',inv)
            #     cv2.waitKey()
            #     print('here')
            cv2.imwrite(outputPath + "/" + c + ".png", crop[index[1][0]:index[1][1], index[0][0]:index[0][1]])
            cv2.imwrite(outputPath + "/" + c + "_final.png", roi)
            cv2.imwrite(outputPath + "/" + c + "_preMask.png", inv)
            cv2.imwrite(outputPath + "/" + c + "_mask.png", hsvImage)
            outputImage.append(outputPath + "/" + c + ".png")
        else:
            outputIntensity.append(math.nan)
            outputPercent.append(0)
            outputImage.append(None)
    # outputPercent = [outputPercent[i] if outputPercent[i] is not None and outputPercent[i] > 0 else 0 for i in range(len(outputPercent))]
    # outputPercent=minmax_scale(outputPercent)
    # outputIntensity=minmax_scale(outputIntensity)
    # outputPercent = (np.round(outputPercent, 2) * 100).astype(np.uint8)
    # dataframe.insert(3,"percent", outputPercent)
    outputIntensity = np.nan_to_num(outputIntensity, nan=np.nanmin(outputIntensity))
    normalized=preprocessing.normalize([outputIntensity])[0]
    # referenceAdded = np.insert(outputIntensity,len(outputIntensity),729976.263)
    l = 1 / 600581.5426
    referenceAdded = 1-1/np.exp(l*outputIntensity)
    # referenceAdded = l/np.exp(l * normalized)
    genOutputIntensity=referenceAdded*100
    # base5=np.vectorize(lambda x :5 * round(x / 5))
    # outputIntensity=base5(outputIntensity)
    # outputIntensity = scipy.stats.zscore(outputIntensity)
    # l = 1/np.mean(outputIntensity)
    # outputIntensity = 1-1/np.exp(l*outputIntensity)
    # outputIntensity = 1/np.exp(outputIntensity)
    # outputIntensity = np.interp(outputIntensity, (outputIntensity.min(), outputIntensity.max()), (0, 100))
    detected = genOutputIntensity > activityThreshold
    dataframe["Detection"] = detected
    dataframe["Intensity"] = genOutputIntensity
    dataframe["normalized"]=normalized
    dataframe["raw_Intensity"] = outputIntensity
    dataframe.to_excel(name[0] + ".xlsx",index=False)
    # dataframe.insert(5,"Image",outputImage)
    dataframe["Image"] = outputImage
    # index = getIndex("1-B6",xpeaks,ypeaks)
    # cv2.imwrite(name[0]+"_1-B6.png",crop[index[1][0]:index[1][1],index[0][0]:index[0][1]])
    for i, p in enumerate(xpeaks):
        if i % 4 == 0:
            cv2.line(crop, (p, 0), (p, crop.shape[1] - 1), (0, 0, 255), thickness=3)
        else:
            cv2.line(crop, (p, 0), (p, crop.shape[1] - 1), (0, 0, 0), thickness=1)
    for i, p in enumerate(ypeaks):
        if i % 4 == 0:
            cv2.line(crop, (0, p), (crop.shape[1] - 1, p), (0, 0, 255), thickness=3)
        else:
            cv2.line(crop, (0, p), (crop.shape[1] - 1, p), (0, 0, 0), thickness=1)
    cv2.imwrite('crop_{}.png'.format(name[0]), crop)
    opHtml = html.format(name[0], css,
                         dataframe.to_html(escape=False, formatters=dict(Image=imageTageGenerator, Intensity=getTruth)))
    file = open('{}.html'.format(name[0]), "w")
    file.write(opHtml)
    file.close()
    dataframes.append(dataframe)
for dataframe in dataframes:
    for intensity,path in zip(dataframe.Intensity,dataframe.Image):
        if path is not None:
            for i in range(10,110,10):
                if intensity<=i:
                    name = path.split("/")
                    copyfile(path,clusterPath+str(i)+"/"+str(round(intensity,2))+"_"+(name[-2]+"_"+name[-1]))
                    break
dataset= pd.concat(dataframes,ignore_index=True)
dataset=dataset.iloc[:,[3,5,6,7]]
dataset.to_excel("dataset.xlsx")


    # plt.figure(111)
    # plt.bar(range(len(yProj)),yGrad)
    #
    # # xGrad = np.gradient(xGrad)
    # # plt.bar(range(len(xProj)),xGrad)
    # plt.figure(131)
    # plt.bar(range(len(yProj)),yProj)
    # plt.show()
    #
    # plt.figure(111)
    # plt.bar(range(len(xProj)),xGrad)
    #
    # # xGrad = np.gradient(xGrad)
    # # plt.bar(range(len(xProj)),xGrad)
    # plt.figure(131)
    # plt.bar(range(len(xProj)),xProj)
    # plt.show()
    # # cv2.waitKey(0)
