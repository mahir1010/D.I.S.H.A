import logging
import os

import bs4 as bs
import cv2
import imutils
import matplotlib.pyplot as plt
import pandas as pd

from Experiment import Experiment, export_experiment
from Image import Image
from SegmentationModel import SegmentationModel
from utils import *

css = '''
body{
      --textC: 245,245,245;
      --backgroundC:35,35,35;
      background: rgb(var(--backgroundC));
      color:rgb(var(--textC));
    }/usr/lib/python3.10/site-packages/setuptools/
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


def checkBoxGenerator(input):
    if input:
        return '<input type="checkbox" checked onchange="updateStatus(this)></input>'
    else:
        return '<input type="checkbox" onchange="updateStatus(this)"></input>'


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


def getLevelRange(img, percentile=0.01, skipZero=False):
    # Assuming grayscale
    histogram = np.histogram(img.flatten(), range(257))
    histogram[0][0] = 0 if skipZero else histogram[0][0]
    cumulativeSum = np.cumsum(histogram[0])
    lower = upper = 255
    for i, j in enumerate(cumulativeSum):
        if i < lower and j >= percentile * cumulativeSum[-1]:
            lower = i
        if upper > i and j >= (1 - percentile) * cumulativeSum[-1]:
            upper = i
            break
    return [lower, upper]


def getPeakBase(img, percent=5):
    histogram = np.histogram(img.flatten(), range(257))[0]
    idx = np.argmax(histogram)
    for i in range(idx - 1, 0, -1):
        if histogram[i] > 0 and histogram[i] <= histogram[idx] * percent / 100:
            return i
    return 0


def createProjectionImages(gray, xProj, yProj, path='./'):
    height, width = gray.shape
    blankImage = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(width):
        cv2.line(blankImage, (row, 0), (row, int(xProj[row])), (255, 255, 255), 1)
    mean = int(np.mean(xProj))
    cv2.line(blankImage, (0, mean), (width, mean), (255, 0, 0), 3)
    cv2.imwrite(path + '_vert.jpg', blankImage)
    blankImage = np.zeros((height, width, 3), dtype=np.uint8)
    for column in range(height):
        cv2.line(blankImage, (0, column),
                 (int(yProj[column]), column), (255, 255, 255), 1)
    mean = int(np.mean(yProj))
    cv2.line(blankImage, (mean, 0), (mean, height), (255, 0, 0), 3)
    cv2.imwrite(path + '_horz.jpg', blankImage)


header = '<th>Intensity</th><th>Area</th><th>Image</th>'

debug = False


def show_image(img, size=1024):
    showimg = imutils.resize(img.copy(), width=size)
    plt.imshow(showimg, cmap='gray')
    plt.show()


def crop_assay(imageObject: Image, cropt_percent=0.1):
    image = imageObject.image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = rescaleIntensity(gray, [60, 255])
    gray = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)[1]
    x, y = getHistogramProjection(gray), getHistogramProjection(gray, axis=1)
    x_crop = [0, 0]
    y_crop = [0, 0]
    for i in range(len(x)):
        if i != 0:
            x_crop[0] = i
            break
    for i in range(len(x), 0, -1):
        if i != 0:
            x_crop[1] = i
            break
    for i in range(len(y)):
        if i != 0:
            y_crop[0] = i
            break
    for i in range(len(y), 0, -1):
        if i != 0:
            y_crop[1] = i
            break
    padding_x = int((x_crop[1] - x_crop[0]) * cropt_percent)
    padding_y = int((y_crop[1] - y_crop[0]) * cropt_percent)
    img_cropped = image[y_crop[0] + padding_y:y_crop[1] - padding_y, x_crop[0] + padding_x:x_crop[1] - padding_x]
    imageObject.image = img_cropped
    cv2.imwrite(f'{imageObject.output_colonies_path}.png', cv2.cvtColor(img_cropped, cv2.COLOR_RGB2BGR))
    return None


def extract_grids(imageObject: Image):
    image = imageObject.image
    # show_image(img_cropped)
    if 0 in image.shape:
        logging.error(f"Image Cropping failed for {imageObject.file_path}")
        raise Exception(f"Image Cropping failed for {imageObject.file_path}")

    # 3. Pre-process image and perform peak detection to find the colonies
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    level_range = getLevelRange(gray)
    gray = rescaleIntensity(gray, [0, level_range[1]])
    # show_image(gray)
    # Rows
    img = cv2.GaussianBlur(gray, (5, 5), 3)
    img = cv2.medianBlur(img, 5)
    img = np.uint8(np.absolute(cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=5)))
    img = cv2.GaussianBlur(img, (5, 5), 3)
    # show_image(img)
    level_range = getLevelRange(img)
    img = rescaleIntensity(img, [sum(level_range) / 2, level_range[1]])
    # show_image(img)
    img = ((img > 90) * 255).astype(np.uint8)
    # show_image(img)
    x_proj = getHistogramProjection(img, 0)
    smooth = 30
    x_proj = np.convolve(x_proj, [1 / smooth for i in range(smooth)])[:-(smooth - 1)]
    x_proj = (x_proj > 50) * x_proj
    x_peaks, x_grad = getPeaks(x_proj, 0, 0, np.mean(x_proj) * 0.6)
    # Columns
    img = cv2.GaussianBlur(gray, (5, 5), 3)
    img = cv2.medianBlur(img, 5)
    img = np.uint8(np.absolute(cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=5)))
    img = cv2.GaussianBlur(img, (5, 5), 3)
    # show_image(img)
    level_range = getLevelRange(img)
    img = rescaleIntensity(img, [sum(level_range) / 2, level_range[1]])
    # show_image(img)
    img = ((img > 90) * 255).astype(np.uint8)
    # show_image(img)
    y_proj = getHistogramProjection(img, 1)
    y_proj = np.convolve(y_proj, [1 / smooth for i in range(smooth)])[:-(smooth - 1)]
    y_proj = (y_proj > 50) * y_proj
    y_peaks, y_grad = getPeaks(y_proj, 0, 0, np.mean(y_proj) * 0.6)
    createProjectionImages(gray, x_proj, y_proj, imageObject.output_colonies_path)
    cv2.imwrite(f'{imageObject.output_colonies_path}.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    createProjectionImages(gray, x_proj, y_proj)
    logging.info(f"For {imageObject.file_path}, {len(x_peaks)},{len(y_peaks)} detected")
    return x_peaks, y_peaks


def export_grid_image(imageObject: Image):
    image = imageObject.image.copy()
    for i, p in enumerate(imageObject.rows):
        if i % 4 == 0:
            cv2.line(image, (p, 0),
                     (p, image.shape[1] - 1), (0, 0, 255), thickness=3)
        else:
            cv2.line(image, (p, 0),
                     (p, image.shape[1] - 1), (0, 0, 0), thickness=3)
    for i, p in enumerate(imageObject.cols):
        if i % 4 == 0:
            cv2.line(image, (0, p),
                     (image.shape[1] - 1, p), (0, 0, 255), thickness=3)
        else:
            cv2.line(image, (0, p),
                     (image.shape[1] - 1, p), (0, 0, 0), thickness=3)
    # cv2.imwrite(os.path.join(imageObject.output_colonies_path, 'grids.png'), image)
    return image


def analyze_image(imageObject: Image, model: SegmentationModel):
    gray = cv2.cvtColor(imageObject.image, cv2.COLOR_RGB2GRAY)
    gray = cv2.bitwise_not(gray)
    mask = model.predict(imageObject.image, 20)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.GaussianBlur(mask, (3, 3), 1.5)
    imageObject.intensity_map = mask
    df = imageObject.dataframe
    outputIntensity = []
    outputArea = []
    outputImage = []
    for c in df['Coordinate']:
        if type(c) is str and (len(c.split("-")[1]) > 1):
            index, colonies = imageObject.getIndex(c)
            roi = mask[index[1][0]:index[1][1], index[0][0]:index[0][1]]
            roi[roi > 0] = 255
            for ix, colony in enumerate(colonies):
                sub_roi = roi[colony[1][0]:colony[1][1], colony[0][0]:colony[0][1]]
                contours, hierarchy = cv2.findContours(sub_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                center = np.divide(sub_roi.shape, 2)
                center_contour = None
                min_distance = sub_roi.size
                threshold_distance = sub_roi.shape[0] // 2
                if len(contours) != 0:
                    for contour in contours:
                        moments = cv2.moments(contour)
                        if moments["m00"] != 0:
                            centroid = [
                                int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])]
                            distance = math.sqrt(np.sum(np.square(centroid - center)))
                            if distance < min_distance and distance < threshold_distance:
                                center_contour = contour
                                min_distance = distance
                    if center_contour is not None:
                        colony_processed = np.zeros_like(sub_roi, dtype=np.uint8)
                        colony_processed = cv2.drawContours(colony_processed, [center_contour], 0, (255, 255, 255),
                                                            cv2.FILLED)
                        roi[colony[1][0]:colony[1][1], colony[0]
                                                       [0]:colony[0][1]] = colony_processed
            gray_roi = gray[index[1][0]:index[1][1], index[0][0]:index[0][1]]
            areaImage = roi.copy()
            gray_roi = cv2.bitwise_and(gray_roi, gray_roi, mask=areaImage)
            areaImage[areaImage > 0] = 1
            area = np.sum(areaImage)
            # intensity = np.divide(intensity, np.subtract(256, intensity))
            intensity = np.sum(gray_roi) / area if area != 0 else 0
            outputIntensity.append(intensity)
            outputArea.append(area)
            outputImage.append(c)
        else:
            outputArea.append(math.nan)
            outputIntensity.append(math.nan)
            outputImage.append(None)
    outputIntensity = np.nan_to_num(outputIntensity, nan=np.nanmin(outputIntensity))
    outputArea = np.nan_to_num(outputArea, nan=np.nanmin(outputArea))
    imageObject.dataframe["Intensity"] = np.round(outputIntensity, 2)
    imageObject.dataframe["Area"] = np.round(outputArea, 2)
    imageObject.dataframe["Image"] = outputImage
    imageObject.raw_dataframe = imageObject.dataframe.copy(deep=True)


def analyze_image_mannual(imageObject: Image):
    crop = imageObject.image
    hsvCrop = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    df = imageObject.dataframe
    outputIntensity = []
    outputArea = []
    outputImage = []
    binaryImage = np.zeros((hsvCrop.shape[0], hsvCrop.shape[1]), np.uint8)
    for c in df["Coordinate"]:
        if type(c) is str and (len(c.split("-")[1]) > 1):
            index, colonies = imageObject.getIndex(c)
            roi = (hsvCrop[index[1][0]:index[1][1], index[0][0]:index[0][1]])[:, :, 2]
            roi = cv2.medianBlur(roi, ksize=3)
            roi = cv2.GaussianBlur(roi, (3, 3), 1.5)
            levelRange = getLevelRange(roi, 0.14)
            levelRange[0] = max(0, levelRange[0] - 20)
            roi = rescaleIntensity(roi, levelRange)
            roi = cv2.GaussianBlur(roi, (5, 5), 2.5)
            lower_bound = getPeakBase(roi, 5)
            roi = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY_INV)[1]
            hsvImage = np.zeros_like(roi, dtype=np.uint8)
            for ix, colony in enumerate(colonies):
                subRoi = roi[colony[1][0]:colony[1][1], colony[0][0]:colony[0][1]]
                if True:
                    contours, hierarchy = cv2.findContours(subRoi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    center = np.divide(subRoi.shape, 2)
                    centerContour = None
                    minDistance = subRoi.size
                    threshold_distance = subRoi.shape[0] // 2
                    if len(contours) != 0:
                        for contour in contours:
                            moments = cv2.moments(contour)
                            if moments["m00"] != 0:
                                centroid = [
                                    int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])]
                                distance = math.sqrt(np.sum(np.square(centroid - center)))
                                if distance < minDistance and distance < threshold_distance:
                                    centerContour = contour
                                    minDistance = distance
                        if centerContour is not None:
                            colonyProcessed = np.zeros_like(subRoi, dtype=np.uint8)
                            colonyProcessed = cv2.drawContours(colonyProcessed, [centerContour], 0, (255, 255, 255),
                                                               cv2.FILLED)
                            if np.sum(colonyProcessed) / 255 < subRoi.size * 0.60:
                                hsvImage[colony[1][0]:colony[1][1], colony[0]
                                                                    [0]:colony[0][1]] = colonyProcessed
            binaryImage[index[1][0]:index[1][1], index[0][0]:index[0][1]] = hsvImage
            inv = (hsvCrop[index[1][0]:index[1][1],
                   index[0][0]:index[0][1]])[:, :, 0]
            roi = cv2.bitwise_and(inv, inv, mask=hsvImage)
            areaImage = roi.copy()
            areaImage[areaImage > 0] = 1
            area = np.sum(areaImage)
            intensity = roi.copy()
            intensity = np.divide(intensity, np.subtract(256, intensity))
            outputIntensity.append(np.sum(intensity))
            outputArea.append(area)
            if debug:
                cv2.imwrite(os.path.join(imageObject.output_colonies_path, c) + "_final.png", roi)
                cv2.imwrite(os.path.join(imageObject.output_colonies_path, c) + "_preMask.png", inv)
                cv2.imwrite(os.path.join(imageObject.output_colonies_path, c) + "_mask.png", hsvImage)
            outputImage.append(os.path.join(imageObject.name.split('.')[0], c + ".png"))
        else:
            outputArea.append(math.nan)
            outputIntensity.append(math.nan)
            outputImage.append(None)
    outputIntensity = np.nan_to_num(
        outputIntensity, nan=np.nanmin(outputIntensity))
    cv2.imwrite(os.path.join(imageObject.output_colonies_path, f'{imageObject.name}_binary_image.png'), binaryImage)
    outputArea = np.nan_to_num(outputArea, nan=-1)
    imageObject.dataframe["Intensity"] = outputIntensity
    imageObject.dataframe["Area"] = outputArea
    imageObject.dataframe["Image"] = outputImage
    imageObject.raw_dataframe = imageObject.dataframe.copy(deep=True)


def convert_bgr_to_rgb(imgObj: Image):
    imgObj.image = cv2.cvtColor(imgObj.image, cv2.COLOR_BGR2RGB)


def process_yeast(experiment: Experiment, normalize=True, export_grid_images=False, export_html=False):
    model = SegmentationModel('efficientnetb1', "Model Weights/")
    global activityThreshold
    output_path = experiment.output_path
    df = experiment.datasheet

    for image, grid in experiment.apply_funct(crop_assay):
        pass

    for image, grid in experiment.apply_funct(extract_grids):
        image.set_grid(*grid)

    if export_grid_images:
        for image, output in experiment.apply_funct(export_grid_image):
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(image.output_colonies_path, f'{image.name}_grid.png'), output)
            pass

    for image, output in experiment.apply_funct(analyze_image, model):
        pass

    # intensities = []
    # area = []
    # plateMedianArea = []
    # plateMedian = []
    # if normalize:
    #     for dataframe in dataframes:
    #         intensities.extend(list(dataframe['Intensity']))
    #         area.extend(list(dataframe['Area']))
    #         plateMedianArea.append(np.median(dataframe['Area']))
    #         plateMedian.append(np.median(dataframe['Intensity']))
    #     median = np.median(plateMedian) + 0.1
    #     aMedian = np.median(plateMedianArea) + 0.1
    #     for dataframe in dataframes:
    #         normalized = np.array(list(dataframe['Intensity'])) / median
    #         normalized = (normalized-normalized.min())/(normalized.max()-normalized.min())*100
    #         dataframe['Intensity'] = normalized.round(3)
    #         normalized = np.array(list(dataframe['Area'])) / aMedian
    #         normalized = (normalized-normalized.min())/(normalized.max()-normalized.min())*100
    #         dataframe['Area'] = normalized.round(3)
    #         # detected = genOutputIntensity > activityThreshold

    experiment.generate_evaluation_table()
    imgs = [image for image in experiment.images if not image.exception_occurred]

    experiment.images = [image for image in experiment.images if not image.exception_occurred]

    export_experiment(experiment)

    # if export_html:
    #     dataframe = pd.concat(dataframes, axis=0)
    #     dataframe = dataframe[['Activated', 'Coordinate', 'TF1', 'TF2', 'Intensity', 'Area', 'Image',
    #                            'tf1empty_intensity', 'tf1empty_area', 'tf1empty_image',
    #                            'tf2empty_intensity', 'tf2empty_area', 'tf2empty_image',
    #                            'empty_empty_intensity', 'empty_empty_area', 'empty_empty_image']]
    #
    #     name = name[:name.rfind('_')]
    #     splt = name.split("_")
    #     splt.pop(2)
    #     name = "_".join(splt)
    #     opHtml = html.format(name, css,
    #                          dataframe.to_html(index=False, header=False, escape=False,
    #                                            formatters=dict(Activated=checkBoxGenerator,
    #                                                            Image=imageTageGenerator,
    #                                                            tf1empty_image=imageTageGenerator,
    #                                                            tf2empty_image=imageTageGenerator,
    #                                                            empty_empty_image=imageTageGenerator)).replace(
    #                              "<tbody>", STATIC_HEADER.format(experiment.bait) + '\n<tbody>'), javascript,
    #                          downloadHTML)
    #     file = open(os.path.join(output_path, f'{name}.html'), "w")
    #     file.write(opHtml)
    #     file.close()

    del experiment

    return 0, 'ok'


def saveExtractedRows(path, extractedRows, root_dir=''):
    extractedRows.sort()
    name = path.split("/")[-1].split('.html')[0]
    path = path[:-1] if path[-1] == '?' else path
    path = path[1:] if path[0] == '/' else path
    path = os.path.join(root_dir, path)
    htmlFile = open(path, 'r').read()
    htmlFile = bs.BeautifulSoup(htmlFile, 'html5lib')
    table = htmlFile.find_all('table')[0]
    allColumns = [header.text for header in table.find('thead').find_all('th')]
    columns = allColumns[allColumns.index('Index'):]
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
    headers.findAll('th')[0]['colspan'] = 2
    headers = str(headers)
    testHTML = testHTML.iloc[extractedRows]
    opHtml = html.format(name, css,
                         testHTML.to_html(header=False, escape=False,
                                          formatters=dict(Image=imageTageGenerator)).replace("<tbody>",
                                                                                             headers + '\n<tbody>'), "",
                         "")
    file = open(path[:-5] + '_Activated.html', "w")
    file.write(opHtml)
    file.close()
    return 0, 'ok'
