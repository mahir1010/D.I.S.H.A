import math
import sys
from pathlib import Path
import bs4 as bs
import cv2
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from shutil import copyfile, rmtree
import sklearn
from sklearn import preprocessing
import glob
import urllib.parse as parse

import os

row2IndexMap = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}

# , index_col=None, header=None)

css = '''
body{
      --textC: 245,245,245;
      --backgroundC:35,35,35;
      background: rgb(var(--backgroundC));
      color:rgb(var(--textC));
    }
    .fixed-nav-bar {
      box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
      direction: ltr;
      height: 3%;
    }

    h1 {
      font-size: 30px;
      text-transform: uppercase;
      font-weight: bolder;
      text-align: center;
      margin-bottom: 15px;
    }

    button {
      border: none;
      box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
      transition: 0.3s;
      border-radius: 5px;
      background-color: #03a9f4;
      transition: 1s;
    }

    table {
      table-layout: fixed;
      border: None;
      width:100%;
      font-size: 18px;
      direction: ltr;
    }

    .fixTableHead {
      overflow-y: auto;
      height: 97%;
      direction: ltr;
    }

    .fixTableHead thead th {
      background:rgba(var(--backgroundC),0.98);
      position: sticky;
      top: 0;
    }

    input[type="checkbox"]:hover {
      box-shadow: 0 8px 16px 0 rgba(0, 0, 0, 0.2);
    }

    th {
      padding: 20px 15px;
      text-align: left;
      font-weight: bolder;
      font-size: 14px;
      text-transform: uppercase;
      text-align-last: center;
      box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
      border: none;
    }

    td:hover {
      box-shadow: 0 8px 16px 0 rgba(0, 0, 0, 0.2);
    }

    td {
      padding: 1px;
      text-align: left;
      vertical-align: middle;
      font-weight: bolder;

      border: none;
      box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
      min-width: 170px;
      text-align: center;
    }


    /* demo styles */
    body {
      font-family: 'Roboto', sans-serif;
      font-weight: bolder;
    }

    section {
      margin: 50px;
    }'''
javascript = '''
<script type="text/javascript">
updateStatus=function(checkbox) {
  if (checkbox.getAttribute("checked")==null){
  checkbox.setAttribute("checked","true");
  }else{
    checkbox.removeAttribute("checked");
  }
}
function post() {
  const form = document.createElement('form');
  form.method = 'post';
  form.action = '/download/extract/';
  inp=document.createElement('input');
  inp.type='text';
  inp.value=decodeURI(window.location.href.split('display')[1]);
  inp.name='title';
  form.appendChild(inp)
  data=document.getElementsByClassName("dataframe")[0];
  rows=data.getElementsByTagName("tr");
  for (var i=0; i<rows.length; i++){
  data = rows[i].getElementsByTagName("td");
  for (var j=0;j<data.length;j++) {
    if (data[j].firstChild.type=="checkbox") {
      data[j].firstChild.name=""+i
      form.appendChild(data[j].firstChild);
    }
  }
  }
  document.body.appendChild(form);
  form.submit();
}
</script>
'''
downloadHTML = '<div><nav class="fixed-nav-bar"><button onclick="post()">Save</button></nav></div>'
html = '''<html><head><title>{0}</title><style>{1}</style>{3}</head><body>{4}<div class="fixTableHead">{2}</div><body></html>'''

### GLOBAL VAR ######
activityThreshold = 0.01


### GLOBAL VAR ######
def checkBoxGenerator(input):
    if input:
        return '<input type="checkbox" checked onchange="updateStatus(this)></input>'
    else:
        return '<input type="checkbox" onchange="updateStatus(this)"></input>'



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
    val = float(val)
    if val < activityThreshold:
        return '<b style="color:red;">' + str(round(val, 3)) + '</b>'
    else:
        return '<b style="color:green;">' + str(round(val, 3)) + '</b>'


def getPeaks(proj, thresh1, thresh2):
    gradient = np.gradient(proj)
    gradient = (np.logical_or(gradient > thresh1,
                              gradient < thresh2)) * gradient
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


def getLevelRange(img, percentile=0.01,skipZero=False):
    # Assuming grayscale
    histogram = np.histogram(img.flatten(), range(257))
    histogram[0][0]=0 if skipZero else histogram[0][0]
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
        cv2.line(blankImage, (0, row),
                 (int(yProj[row]), row), (255, 255, 255), 1)
    # cv2.imwrite(path + 'Horz.jpg', blankImage)
    blankImage = np.zeros((height, width, 3), dtype=np.uint8)
    for column in range(width):
        cv2.line(blankImage, (column, 0),
                 (column, int(xProj[column])), (255, 255, 255), 1)
    # cv2.imwrite(path + 'vert.jpg', blankImage)

def addRows(dataframe):
    empty=dataframe.loc[(dataframe['TF2'] == "empty")&(dataframe['TF1'] == "empty")].iloc[0]
    targetDf=dataframe.loc[(dataframe['TF2'] != "empty")&(dataframe['TF1'] != "empty")].copy(deep=True)
    for index,row in dataframe.iterrows():
        targetDf.loc[index+0.2]=dataframe.loc[(dataframe['TF2'] == "empty")&(dataframe['TF1'] == row['TF1'])].iloc[0]
        targetDf.loc[index + 0.4] = dataframe.loc[(dataframe['TF1'] == "empty") & (dataframe['TF2'] == row['TF2'])].iloc[0]
        targetDf.loc[index + 0.6] = empty
    targetDf.sort_index(inplace=True)
    targetDf.reset_index(drop=True,inplace=True)
    return targetDf

header='<th>Intensity</th><th>Area</th><th>Image</th>'

def process_yeast(dir_path, excels_path, template1, template2, debug=False, extensions=['[jJ][pP][gG]','[pP][nN][gG]']):
    global activityThreshold
    # if len(sys.argv) == 1:
    #     print("Usage test.py <path_to_images>")
    #     exit(-1)
    path=[]
    output_path=os.path.join(dir_path,'output')
    for ext in extensions:
        path.extend(glob.glob(os.path.join(dir_path, "*.{}".format(ext))))  # glob.glob(sys.argv[1] + "/*.JPG")
    path.sort()
    df = pd.read_excel(excels_path, engine='openpyxl')
    # clusterPaths=[]
    baits={}
    for file in path:
        name=Path(file).stem.split('.')[0]
        properties=name.split('_')
        baitNo=properties[1].split('-')[-1]
        if baitNo not in baits:
            baits[baitNo]=[]
        baits[baitNo].append(file)

    for bait in baits.keys():
        headers='<thead><tr>'+'<th colspan="3">Bait Number {}</th>'.format(bait)+'<th colspan="2">TF</th>'+''.join(['<th colspan="3">Day {}</th>'.format(name.split('_')[-1][0]) for name in baits[bait]])+'</tr>'+'<tr><th>Index</th><th>Activated</th><th>Coordinate</th><th>TF1</th><th>TF2</th>' + (header * len(baits[bait])) + '</tr></thead>'
        dataframes = []
        outputPaths = []
        fileNames = []
        for imagePath in baits[bait]:
            name = Path(imagePath).stem
            name = name.split('.')
            fileNames.append(name)
            outputPath = os.path.join(output_path, name[0])  # "./output/{}".format(name[0])
            Path(outputPath).mkdir(parents=True, exist_ok=True)
            outputPaths.append(outputPath)
            # clusterPath = os.path.join(output_path, "cluster/")  # "./output/cluster/"
            # clusterPaths.append(clusterPath)
            # rmtree(clusterPath, ignore_errors=True)
            # for i in range(10, 110, 10):
            #     Path(clusterPath + str(i)).mkdir(parents=True, exist_ok=True)
            # dataframe = pd.read_excel(os.path.join(output_path, name[
            #     0] + ".xlsx"), engine = 'openpyxl')  # pd.read_excel(name[0] + ".xlsx", engine='openpyxl')
            dataframe = df.copy(deep=True)
            # print(list(dataframe.columns))
            image = cv2.imread(imagePath)
            if image is None:
                return -1, "Can't access image at " + dir_path
            img = cv2.Canny(image, 20, 30)
            # cv2.imwrite(os.path.join(output_path, 'edge-detect.jpg'), img)#'edge-detect.jpg', img)
            # template1 = cv2.imread(os.path.join(output_path,'upperLeft.png'), 0)
            # template2 = cv2.imread(os.path.join(output_path,'bottomRight.png'), 0)

            res = cv2.matchTemplate(img, template1, cv2.TM_CCORR_NORMED)
            res1 = cv2.matchTemplate(img, template2, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = np.add(max_loc, [140, 40])
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res1)
            bottom_right = np.add(max_loc, [0, 140])
            img = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            # cv2.imwrite(os.path.join(output_path, 'first crop.png'), img)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            levelRange = getLevelRange(gray)
            gray = rescaleIntensity(gray, [0, levelRange[1]])
            gray = cv2.Sobel(gray, cv2.CV_8UC1, 1, 0, ksize=5)
            gray = cv2.medianBlur(gray, 5)
            gray = cv2.GaussianBlur(gray, (5, 5), 3)
            levelRange = getLevelRange(gray)
            gray = rescaleIntensity(gray, [sum(levelRange) / 2, levelRange[1]])
            gray = ((gray > 90) * 255).astype(np.uint8)
            # cv2.imwrite(os.path.join(output_path, 'gray.jpg'), gray)
            xProj, yProj = getHistogramProjection(gray)
            smooth = 5
            xProj = np.convolve(
                xProj, [1 / smooth for i in range(smooth)])[:-(smooth - 1)]
            yProj = np.convolve(
                yProj, [1 / smooth for i in range(smooth)])[:-(smooth - 1)]
            xProj = (xProj > 20) * xProj
            yProj = (yProj > 20) * yProj

            xNonZero = [i for i, elem in enumerate(xProj) if elem > 30]
            yNonZero = [i for i, elem in enumerate(yProj) if elem > 30]
            crop = img
            xpeaks, xGrad = getPeaks(xProj, 0, 0)

            ypeaks, yGrad = getPeaks(yProj, 0, 0)
            outputPercent = []
            outputIntensity = []
            outputArea=[]
            hsvCrop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            outputImage = []
            # reference, refCols = getIndex("3-B7", xpeaks, ypeaks)

            for c in df["Coordinate"]:
                if (len(c.split("-")[1]) > 1):
                    index, colonies = getIndex(c, xpeaks, ypeaks)
                    roi = (hsvCrop[index[1][0]:index[1][1],
                           index[0][0]:index[0][1]])[:, :, 2]
                    hsvImage = np.zeros_like(roi, dtype=np.uint8)
                    for ix, colony in enumerate(colonies):
                        subRoi = roi[colony[1][0]:colony[1]
                        [1], colony[0][0]:colony[0][1]]
                        levelRange = getLevelRange(subRoi)
                        if True:
                            colonyProcessed = rescaleIntensity(
                                subRoi, [0, levelRange[1]])
                            colonyProcessed = cv2.GaussianBlur(
                                colonyProcessed, (3, 3), 2.5)
                            colonyProcessed = cv2.medianBlur(colonyProcessed,ksize=9)
                            test = getLevelRange(colonyProcessed, 0.05)
                            # if test[1] - test[0] <= 127:
                            #     test[0] = 127
                            colonyProcessed = cv2.threshold(
                                colonyProcessed, test[0], 255, cv2.THRESH_BINARY_INV)[1]
                            contours, hierarchy = cv2.findContours(
                                colonyProcessed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            center = np.divide(colonyProcessed.shape, 2)
                            centerContour = None
                            minDistance = colonyProcessed.size
                            for contour in contours:
                                moments = cv2.moments(contour)
                                if moments["m00"] != 0:
                                    centroid = [
                                        int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])]
                                    distance = math.sqrt(
                                        np.sum(np.square(centroid - center)))
                                    if distance < minDistance:
                                        centerContour = contour
                                        minDistance = distance
                            colonyProcessed = np.zeros_like(
                                colonyProcessed, dtype=np.uint8)

                            colonyProcessed = cv2.drawContours(
                                colonyProcessed, [centerContour], 0, (255, 255, 255), cv2.FILLED)
                            hsvImage[colony[1][0]:colony[1][1], colony[0]
                                                                [0]:colony[0][1]] = colonyProcessed

                    inv = (hsvCrop[index[1][0]:index[1][1],
                           index[0][0]:index[0][1]])[:, :, 0]
                    inv = rescaleIntensity(inv, [getLevelRange(inv)[0], 255])
                    invRange = getLevelRange(inv, 0)
                    if invRange[1] - invRange[0] < 3:
                        inv = np.zeros_like(inv, dtype=np.uint8)
                    roi = cv2.bitwise_and(inv, inv, mask=hsvImage)
                    if c=='1-B5':
                        print('')
                    roi[roi <= getLevelRange(roi,skipZero=True,percentile=.10)[0]] = 0
                    areaImage = roi.copy()
                    areaImage[areaImage > 0] = 1
                    area = np.sum(areaImage)
                    intensity = roi.copy()
                    intensity = np.divide(intensity, np.subtract(256, intensity))
                    outputIntensity.append(np.sum(intensity))
                    outputArea.append(area)
                    cv2.imwrite(os.path.join(outputPath, c + ".png"),  # outputPath + "/" + c + ".png",
                                crop[index[1][0]:index[1][1], index[0][0]:index[0][1]])
                    if debug:
                        cv2.imwrite(os.path.join(outputPath, c) + "_final.png", roi)
                        cv2.imwrite(os.path.join(outputPath, c) + "_preMask.png", inv)
                        cv2.imwrite(os.path.join(outputPath, c) + "_mask.png", hsvImage)
                    outputImage.append(os.path.join(name[0], c + ".png"))
                else:
                    outputArea.append(math.nan)
                    outputIntensity.append(math.nan)
                    outputPercent.append(0)
                    outputImage.append(None)
            outputIntensity = np.nan_to_num(
                outputIntensity, nan=np.nanmin(outputIntensity))
            outputArea=np.nan_to_num(outputArea,nan=-1)
            dataframe["Intensity"] = outputIntensity
            dataframe["Area"]=outputArea
            dataframe["Image"] = outputImage
            for i, p in enumerate(xpeaks):
                if i % 4 == 0:
                    cv2.line(crop, (p, 0),
                             (p, crop.shape[1] - 1), (0, 0, 255), thickness=3)
                else:
                    cv2.line(crop, (p, 0),
                             (p, crop.shape[1] - 1), (0, 0, 0), thickness=1)
            for i, p in enumerate(ypeaks):
                if i % 4 == 0:
                    cv2.line(crop, (0, p),
                             (crop.shape[1] - 1, p), (0, 0, 255), thickness=3)
                else:
                    cv2.line(crop, (0, p),
                             (crop.shape[1] - 1, p), (0, 0, 0), thickness=1)
            cv2.imwrite(os.path.join(output_path, name[0] + '_crop.png'), crop)
            dataframes.append(dataframe)
        intensities = []
        area=[]
        plateMedianArea=[]
        plateMedian = []
        for dataframe in dataframes:
            intensities.extend(list(dataframe['Intensity']))
            area.extend(list(dataframe['Area']))
            plateMedianArea.append(np.median(dataframe['Area']))
            plateMedian.append(np.median(dataframe['Intensity']))
        median = np.median(plateMedian)
        aMedian = np.median(plateMedianArea)
        for dataframe in dataframes:
            normalized = np.array(list(dataframe['Intensity'])) / median
            # normalized = stats.zscore(normalized)
            dataframe['Intensity'] = normalized.round(3)
            normalized = np.array(list(dataframe['Area'])) / aMedian
            # normalized = stats.zscore(normalized)
            dataframe['Area']=normalized.round(3)
            # detected = genOutputIntensity > activityThreshold


        for dataframe, name in zip(dataframes, fileNames):
            excelColumns = list(dataframe.columns)
            excelColumns.remove('Image')
            dataframe[excelColumns].to_excel(os.path.join(output_path, name[0]) + ".xlsx", index=False)
        customDF=[dataframes[0][['Coordinate','TF1','TF2','Intensity','Area','Image']]]
        if len(dataframe)>1:
            customDF.extend(dataframe[['Intensity','Area','Image']] for dataframe in dataframes[1:])
        dataframe=pd.concat(customDF,axis=1)
        #Insert Additional rows
        dataframe=addRows(dataframe)
        #end
        dataframe.insert(0, 'Activated', [False for i in range(len(dataframe.index))])

            # ################################################################
            # dataframe = main_add_cols(dataframe)  # COSMIN
            # ################################################################
            # activityThreshold = dataframe['EMP_EMP_Intensity'][0]
            # dataframe.insert(0, 'Activated', [False for i in range(len(dataframe.index))])
        opHtml = html.format(name[0], css,
                             dataframe.to_html(header=False,escape=False,formatters=dict(Activated=checkBoxGenerator, Image=imageTageGenerator)).replace("<tbody>",headers+'\n<tbody>'), javascript,downloadHTML)
        file = open(os.path.join(output_path, '{}.html'.format(name[0][:name[0].rfind('_')])), "w")
        file.write(opHtml)
        file.close()

    return 0, 'ok'


def saveExtractedRows(path, extractedRows,root_dir=''):
    extractedRows.sort()
    name = path.split("/")[-1].split('.html')[0]
    path = path[:-1] if path[-1] == '?' else path
    path = path[1:] if path[0] == '/' else path
    path = os.path.join(root_dir, path)
    htmlFile = open(path, 'r').read()
    htmlFile = bs.BeautifulSoup(htmlFile, 'html5lib')
    table = htmlFile.find_all('table')[0]
    allColumns = [header.text for header in table.find('thead').find_all('th')]
    columns=allColumns[allColumns.index('Index'):]
    rows = table.find('tbody').find_all('tr')
    data = []
    for row in rows:
        drow = []
        for cell in row:
            if type(cell) != bs.element.Tag:
                continue
            if cell.find('img'):
                drow.append(cell.find('img')['src'])
            elif cell.find('input'):
                drow.append(cell.find('input').has_attr("checked"))
            else:
                drow.append(cell.text)
        drow = [d.text if not d.find('img') else d.find('img')['src'] for d in row.find_all('td')]
        if len(drow) != 0:
            data.append(drow)
    testHTML = pd.DataFrame(data, columns=columns[1:])
    testHTML.drop('Activated', axis=1, inplace=True)
    table.find(text='Activated').parent.extract()
    headers = table.find('thead')
    headers.findAll('th')[0]['colspan']=2
    headers=str(headers)
    testHTML = testHTML.iloc[extractedRows]
    opHtml = html.format(name, css,
                         testHTML.to_html(header=False,escape=False,
                                          formatters=dict(Image=imageTageGenerator)).replace("<tbody>",headers+'\n<tbody>'), "", "")
    file = open(path[:-5] + '_Activated.html', "w")
    file.write(opHtml)
    file.close()
    return 0, 'ok'