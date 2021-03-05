import math
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
from pathlib import Path
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from pretty_html_table import build_table
row2IndexMap={'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
excel_file = "./data/HetY1H Pilot TF coordinates.xlsx"
df = pd.read_excel(excel_file, engine='openpyxl')#, index_col=None, header=None)
css='''
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

def getIndex(inp,rows,cols):
    index=inp.split("-")

    yIndex=row2IndexMap[index[1][0]]
    xIndex=int(index[1][1:])-1
    if index[0]=="1":
        xIndex=4*xIndex
        yIndex=4 * yIndex
    elif index[0]=="2":
        xIndex = 4 * xIndex+2
        yIndex = 4 * yIndex
    elif index[0]=="3":
        xIndex = 4 * xIndex
        yIndex = 4 * yIndex+2
    elif index[0]=="4":
        xIndex = 4 * xIndex+2
        yIndex = 4 * yIndex+2

    return [[rows[xIndex],rows[xIndex+2]],[cols[yIndex],cols[yIndex+2]]]

import glob
def rescaleIntensity(img,val):
    table = np.interp(np.arange(256), [0, val], [0, 255]).astype('uint8')
    return cv2.LUT(img, table),table
def getHistogramProjection(img):
    histogram = img.copy()
    histogram[histogram == 255] = 1
    return np.sum(histogram,axis=0),np.sum(histogram,axis=1)
def imageTageGenerator(path):
    return '''<img src="{}">'''.format(path)
def genPercentage(val):
    return str(val)+"%"
def getPeaks(proj,thresh1,thresh2):
    gradient = np.gradient(proj)
    gradient = (np.logical_or(gradient > thresh1, gradient < thresh2)) * gradient
    peaks = []
    isPeak = False
    lastPeak=0
    for i in range(1, len(gradient)):
        if gradient[i] > 0:
            isPeak = True
        if gradient[i] < 0 and isPeak and (i-lastPeak)>70 and proj[i] > 100:
            peaks.append(i)
            lastPeak=i
            isPeak = False
    diff = np.diff(peaks)
    avg = int(np.mean(diff))
    for i, v in enumerate(diff):
        if math.fabs(v - 2 * avg) < 5:
            peaks.insert(i + 1, peaks[i] + avg)
    peaks.insert(len(peaks),peaks[-1] + avg)
    peaks = np.subtract(peaks,int(avg/2))
    return peaks,gradient

if len(sys.argv)==1:
    print("Usage test.py <path_to_images>")
    exit(-1)
p = sys.argv[1]
p = p[:-1] if p[-1]=="/" else p
path= glob.glob(sys.argv[1]+"/*.JPG")
for imagePath in path:
    dataframe=df.copy()
    image = cv2.imread(imagePath)
    if image is None:
        print("Can't access image at ",sys.argv[1])
        exit(-1)
    img = cv2.Canny(image,20,30)
    cv2.imwrite('edge-detect.jpg',img)
    template1 = cv2.imread('upperLeft.png',0)
    template2 = cv2.imread('bottomRight.png',0)

    res = cv2.matchTemplate(img,template1,cv2.TM_CCORR_NORMED)
    res1= cv2.matchTemplate(img,template2,cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = np.add(max_loc,[140,40])
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res1)
    bottom_right = np.add(max_loc,[0,140])
    img = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    cv2.imwrite('first crop.png',img)
    # cv2.waitKey()
    # cv2.rectangle(img,tuple(top_left), tuple(bottom_right), 255, 2)
    # cv2.imwrite('crop.jpg',img1[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]])
    # cv2.waitKey(0)
    # print(top_left,bottom_right)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.Sobel(gray,cv2.CV_8UC1,1,0,ksize=5)
    gray = cv2.blur(gray,(5,5))
    # hist,bins = np.histogram(img.ravel(),256,[0,256])
    gray = ((gray >90) * 255).astype(np.uint8)
    cv2.imwrite('gray.jpg',gray)
    # cv2.waitKey()
    xProj,yProj=getHistogramProjection(gray)
    smooth=5
    xProj = np.convolve(xProj,[1/smooth for i in range(smooth)])[:-(smooth-1)]
    yProj = np.convolve(yProj,[1/smooth for i in range(smooth)])[:-(smooth-1)]
    xProj = (xProj>20)*xProj
    yProj = (yProj>20)*yProj
    # for i in range(len(xProj)-2):
    #     xProj[i]= (xProj[i]+xProj[i+1]+xProj[i+2])/3
    height,width= gray.shape
    blankImage = np.zeros((height,width,3),dtype=np.uint8)
    for row in range(height):
        cv2.line(blankImage, (0,row), (int(yProj[row]),row), (255,255,255), 1)
    cv2.imwrite('Horz.jpg',blankImage)
    blankImage = np.zeros((height,width,3),dtype=np.uint8)
    for column in range(width):
        cv2.line(blankImage, (column,0), (column,int(xProj[column])), (255,255,255), 1)
    cv2.imwrite('vert.jpg',blankImage)
    xNonZero = [i for i, elem in enumerate(xProj) if elem >30]
    yNonZero= [i for i, elem in enumerate(yProj) if elem >30]
    # crop = img[yNonZero[0]:yNonZero[-1],xNonZero[0]:xNonZero[-1]]
    crop=img
    name = Path(imagePath).stem
    name = name.split('.')
    outputPath="./output/{}".format(name[0])
    xpeaks,xGrad = getPeaks(xProj,0,0)

    ypeaks,yGrad =getPeaks(yProj,0,0)
    outputPercent=[]
    outputIntensity=[]
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    hsvCrop = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    hist1=np.histogram(hsvCrop[:,:,2],range(257))
    hist2= np.histogram(crop[:,:,2],range(257))
    thresh=np.argmax(hist1[0])
    hsvImage,t1 =  rescaleIntensity(hsvCrop[:,:,2],thresh-1)
    hsvImage = cv2.bitwise_not(hsvImage)
    thresh1=np.argmax(hist2[0])
    rgbImage,t2 = rescaleIntensity(crop[:, :, 2], thresh1 - 1)
    rgbImage =cv2.bitwise_not(rgbImage)
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
    cv2.imwrite(name[0]+"_hsv"+str(thresh)+".png",hsvImage)
    cv2.imwrite(name[0]+"_r"+str(thresh1)+".png",rgbImage)
    outputImage=[]
    Path(outputPath).mkdir(parents=True, exist_ok=True)
    reference = getIndex("2-G10",xpeaks,ypeaks)
    reference = np.mean((hsvImage[reference[1][0]:reference[1][1],reference[0][0]:reference[0][1]]))
    for c in dataframe["Coordinate"]:
        if(len(c.split("-")[1])>1):
            index= getIndex(c,xpeaks,ypeaks)
            outputIntensity.append(np.mean((hsvImage[index[1][0]:index[1][1], index[0][0]:index[0][1]])))
            outputPercent.append(np.mean((hsvImage[index[1][0]:index[1][1], index[0][0]:index[0][1]])) - reference)
            cv2.imwrite(outputPath+"/"+c+".png",crop[index[1][0]:index[1][1],index[0][0]:index[0][1]])
            outputImage.append(outputPath+"/"+c+".png")
        else:
            outputIntensity.append(0)
            outputPercent.append(0)
            outputImage.append(None)
    # outputPercent = [outputPercent[i] if outputPercent[i] is not None and outputPercent[i] > 0 else 0 for i in range(len(outputPercent))]
    # outputPercent=minmax_scale(outputPercent)
    # outputIntensity=minmax_scale(outputIntensity)
    # outputPercent = (np.round(outputPercent, 2) * 100).astype(np.uint8)
    # dataframe.insert(3,"percent", outputPercent)
    dataframe.insert(3,"Intesity",outputIntensity)
    dataframe.insert(4,"Image",outputImage)
    # index = getIndex("1-B6",xpeaks,ypeaks)
    # cv2.imwrite(name[0]+"_1-B6.png",crop[index[1][0]:index[1][1],index[0][0]:index[0][1]])
    for i,p in enumerate(xpeaks):
        if i%4==0:
            cv2.line(crop,(p,0),(p,crop.shape[1]-1),(0,0,255),thickness=3)
        else:
            cv2.line(crop, (p, 0), (p, crop.shape[1] - 1), (0, 0, 0), thickness=1)
    for i,p in enumerate(ypeaks):
        if i%4==0:
            cv2.line(crop,(0,p),(crop.shape[1]-1,p),(0,0,255),thickness=3)
        else:
            cv2.line(crop, (0, p), (crop.shape[1] - 1, p), (0, 0, 0), thickness=1)
    cv2.imwrite('crop_{}.png'.format(name[0]),crop)
    opHtml=html.format(name[0],css,dataframe.to_html(escape=False,formatters=dict(Image=imageTageGenerator)))
    file = open('{}.html'.format(name[0]),"w")
    file.write(opHtml)
    file.close()
    # dataframe.to_excel(name[0]+".xlsx")

    # plt.figure(111)
    # plt.bar(range(len(yProj)),yGrad)
    #
    # # xGrad = np.gradient(xGrad)
    # # plt.bar(range(len(xProj)),xGrad)
    # plt.figure(131)
    # plt.bar(range(len(yProj)),yProj)
    # plt.show()

    # plt.figure(111)
    # plt.bar(range(len(xProj)),xGrad)
    #
    # # xGrad = np.gradient(xGrad)
    # # plt.bar(range(len(xProj)),xGrad)
    # plt.figure(131)
    # plt.bar(range(len(xProj)),xProj)
    # plt.show()
    # cv2.waitKey(0)