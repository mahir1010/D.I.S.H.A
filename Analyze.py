import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
from pathlib import Path

def getHistogramProjection(img):
    histogram = img.copy()
    histogram[histogram == 255] = 1
    return np.sum(histogram,axis=0),np.sum(histogram,axis=1)

if len(sys.argv)==1:
    print("Usage test.py <path_to_image>")
    exit(-1)
image = cv2.imread(sys.argv[1])
if image is None:
    print("Can't access image at ",sys.argv[1])
    exit(-1)
img = cv2.Canny(image,20,30)
template1 = cv2.imread('upperLeft.png',0)
template2 = cv2.imread('bottomRight.png',0)

res = cv2.matchTemplate(img,template1,cv2.TM_CCORR_NORMED)
res1= cv2.matchTemplate(img,template2,cv2.TM_CCORR_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = np.add(max_loc,[140,40])
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res1)
bottom_right = np.add(max_loc,[0,140])
img = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
# cv2.imshow('first crop.png',img)
# cv2.waitKey()
# cv2.rectangle(img,tuple(top_left), tuple(bottom_right), 255, 2)
# cv2.imwrite('crop.jpg',img1[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]])
# cv2.waitKey(0)
# print(top_left,bottom_right)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# hist,bins = np.histogram(img.ravel(),256,[0,256])
gray = ((gray < 110) * 255).astype(np.uint8)
# cv2.imshow('gray',gray)
# cv2.waitKey()
xProj,yProj=getHistogramProjection(gray)
xNonZero = [i for i, elem in enumerate(xProj) if elem >30]
yNonZero= [i for i, elem in enumerate(yProj) if elem >30]
crop = img[yNonZero[0]:yNonZero[-1],xNonZero[0]:xNonZero[-1]]
name = Path(sys.argv[1]).stem
name = name.split('.')
cv2.imwrite('crop_{}.png'.format(name[0]),crop)
# cv2.waitKey(0)
