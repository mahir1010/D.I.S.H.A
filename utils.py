import math

import numpy as np

row2IndexMap = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11}
Index2rowMap = {v: k for k, v in row2IndexMap.items()}

STATIC_HEADER = '<thead><tr>' + '<th colspan="4">Bait Number {0}</th>' + '<th colspan="3">TF1-TF2</th>' + \
                '<th colspan="3">TF1-Empty</th>' + '<th colspan="3">TF2-Empty</th>' + '<th colspan="3">Empty-Empty</th>' \
                + '</tr>' + '<tr><th>Activated</th><th>Coordinate</th><th>TF1</th><th>TF2</th>' \
                + '<th>Intensity</th><th>Area</th><th>Image</th>' * 4 + '</tr></thead>'


def extract_bait_number(file_name):
    return file_name.split("_")[1].split("-")[-1]


def extract_plate_name(file_name):
    return file_name.split("_")[2]


def rescaleIntensity(img, range):
    from cv2 import LUT
    table = np.interp(np.arange(256), range, [0, 255]).astype('uint8')
    return LUT(img, table)


def getHistogramProjection(img, axis=0):
    histogram = img.copy()
    histogram[histogram == 255] = 1
    return np.sum(histogram, axis=axis)


def getPeaks(proj, thresh1, thresh2, peak_thresh=100, raw=False):
    gradient = np.gradient(proj)
    gradient = (np.logical_or(gradient > thresh1, gradient < thresh2)) * gradient
    peaks = []
    isPeak = False
    peakStart = 0
    lastPeak = 0
    for i in range(1, len(gradient)):
        if gradient[i] > 0:
            isPeak = True
        if gradient[i] < 0 and isPeak and proj[i] > peak_thresh and (i - lastPeak) > 70:
            peaks.append(i)
            lastPeak = i
            isPeak = False
    if raw:
        return peaks, gradient
    diff = np.diff(peaks)
    avg = int(np.median(diff))
    i = 0
    while i + 1 < len(peaks):
        if round(math.fabs(peaks[i + 1] - peaks[i]) / avg) > 1:
            peaks.insert(i + 1, peaks[i] + avg)
        i += 1
    diff = np.diff(peaks)
    avg = int(np.mean(diff))
    # for i, v in enumerate(diff):
    #     if math.fabs(v - 2 * avg) < 5:
    #         peaks.insert(i + 1, peaks[i] + avg)
    while math.fabs(len(proj) - peaks[-1]) > avg:
        peaks.insert(len(peaks), peaks[-1] + avg)
    peaks.insert(len(peaks), peaks[-1] + avg)
    peaks = np.subtract(peaks, int(avg * 0.75))
    return peaks, gradient


def getWidths(proj):
    pos = []
    width = []
    isNonZero = False
    start = 0
    for i in range(len(proj)):
        if proj[i] > 0 and not isNonZero:
            isNonZero = True
            pos.append(i)
        if proj[i] == 0 and isNonZero:
            isNonZero = False
            width.append(i - pos[-1])
    return pos, width
