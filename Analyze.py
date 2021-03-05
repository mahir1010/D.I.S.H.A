import glob
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale


def getHistogramProjection(img):
    histogram = img.copy()
    histogram[histogram == 255] = 1
    return np.sum(histogram, axis=0), np.sum(histogram, axis=1)


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


def extractEssay(image, topLeft, bottomRight):
    img = cv2.Canny(image, 20, 30)
    tLeftCorner = cv2.matchTemplate(img, topLeft, cv2.TM_CCORR_NORMED)
    bRightCorner = cv2.matchTemplate(img, bottomRight, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(tLeftCorner)
    top_left = np.add(max_loc, topLeft.shape)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(bRightCorner)
    bottom_right = np.subtract(max_loc, bottomRight.shape)
    return image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]


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
    return [[rows[xIndex], rows[xIndex + 2]], [cols[yIndex], cols[yIndex + 2]]]


row2IndexMap = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}


def drawGrids(crop, xpeaks, ypeaks):
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
    return crop


excel_file = "./data/HetY1H Pilot TF coordinates.xlsx"
smooth = 5

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Usage test.py <path_to_images>")
        exit(-1)
    p = sys.argv[1]
    p = p[:-1] if p[-1] == "/" else p
    df = pd.read_excel(excel_file, engine='openpyxl')
    path = glob.glob(sys.argv[1] + "/*.JPG")
    template1 = cv2.imread('upperLeft.png', 0)
    template2 = cv2.imread('bottomRight.png', 0)
    for imagePath in path:
        name = Path(imagePath).stem.split('.')
        image = cv2.imread(imagePath)
        dataframe = df.copy()
        cropped = extractEssay(image, template1, template2)
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        gray = cv2.Sobel(gray, cv2.CV_8UC1, 1, 0, ksize=5)
        gray = cv2.blur(gray, (5, 5))
        gray = ((gray > 90) * 255).astype(np.uint8)
        xProj, yProj = getHistogramProjection(gray)
        xProj = np.convolve(xProj, [1 / smooth for i in range(smooth)])[:-(smooth - 1)]
        yProj = np.convolve(yProj, [1 / smooth for i in range(smooth)])[:-(smooth - 1)]
        xProj = (xProj > 20) * xProj
        yProj = (yProj > 20) * yProj
        xpeaks, xGrad = getPeaks(xProj, 0, 0)
        ypeaks, yGrad = getPeaks(yProj, 0, 0)
        output = []
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        for c in dataframe["Coordinate"]:
            if (len(c.split("-")[1]) > 1):
                index = getIndex(c, xpeaks, ypeaks)
                output.append(np.mean(cropped[index[1][0]:index[1][1], index[0][0]:index[0][1]]))
            else:
                output.append(None)
        output = minmax_scale(output)
        dataframe.insert(3, "Intensity", output)
        dataframe.to_excel(name[0] + ".xlsx")
