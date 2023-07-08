import math
import os
import re

import numpy as np

row2IndexMap = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11}
Index2rowMap = {v: k for k, v in row2IndexMap.items()}

STATIC_HEADER = '<thead><tr>' + '<th colspan="4">Bait Number {0}</th>' + '<th colspan="3">TF1-TF2</th>' + \
                '<th colspan="3">TF1-Empty</th>' + '<th colspan="3">TF2-Empty</th>' + '<th colspan="3">Empty-Empty</th>' \
                + '</tr>' + '<tr><th>Activated</th><th>Coordinate</th><th>TF1</th><th>TF2</th>' \
                + '<th>Intensity</th><th>Area</th><th>Image</th>' * 4 + '</tr></thead>'

name_regex = re.compile(r"[a-zA-Z0-9-]+_[a-zA-Z0-9]+\-\d+_\d+-\d+_[\w-]+")
extensions = ['[jJ][pP][gG]', '[pP][nN][gG]']
coordinate_regex = re.compile(r'\d\d-[a-zA-Z]{1}\d\d')


def verify_image_name(image_name):
    # Expects complete path to the image
    return name_regex.fullmatch(os.path.split(image_name)[1].split('.')[0])


def extract_bait_number(file_name):
    return file_name.split("_")[1].split("-")[-1]


def extract_plate_name(file_name):
    return file_name.split("_")[2]


def rescale_intensity(img, range):
    from cv2 import LUT
    table = np.interp(np.arange(256), range, [0, 255]).astype('uint8')
    return LUT(img, table)


def get_histogram_projection(img, axis=0):
    histogram = img.copy() / 255
    # histogram[histogram == 255] = 1
    return np.sum(histogram, axis=axis)


class Peak:
    def __init__(self, height, begin, center, end):
        self.peak_height = height
        self.peak_begin = begin
        self.peak_center = center
        self.peak_end = end

    @property
    def height(self):
        return self.peak_height

    @property
    def drop(self):
        return self.peak_end - self.peak_center

    @property
    def rise(self):
        return self.peak_center - self.peak_begin

    @property
    def width(self):
        return self.peak_end - self.peak_begin

    def __le__(self, other):
        return self.peak_center <= other

    def __lt__(self, other):
        return self.peak_center < other

    def __gt__(self, other):
        return self.peak_center > other

    def __ge__(self, other):
        return self.peak_center >= other

    def __str__(self):
        return f'c: {self.peak_center} w: {self.width} h: {self.peak_height}'


def get_peaks(proj, thresh_min, thresh_max, min_height_thresh, raw=False):
    gradient = np.gradient(proj)
    gradient = (np.logical_or(gradient > thresh_min, gradient < thresh_max)) * gradient
    peaks = []
    isPeak = False
    lastPeak = 0
    for i in range(1, len(gradient)):
        if gradient[i] > 0:
            isPeak = True
        if gradient[i] < 0 and isPeak and proj[i] > min_height_thresh and (i - lastPeak) > 70:
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
    # while math.fabs(len(proj) - peaks[-1]) > avg:
    #         peaks.insert(len(peaks), peaks[-1] + avg)
    peaks.insert(len(peaks), peaks[-1] + avg)
    peaks = np.subtract(peaks, int(avg * 0.75))
    return peaks, gradient


def get_peaks_v2(proj, min_height_thresh):
    gradient = np.gradient(proj)
    peaks = []
    is_peak = False
    peak_begin = 0
    for i in range(1, len(gradient)):
        if gradient[i] > 0 and not is_peak:
            if len(peaks):
                peaks[-1].peak_end = i
            peak_begin = i
            is_peak = True
        if gradient[i] < 0 and is_peak and proj[i] > min_height_thresh:
            peaks.append(Peak(proj[i], peak_begin, i, -1))
            is_peak = False
    diff = np.diff([p.peak_center for p in peaks])
    avg = int(np.median(diff))
    # TODO Complete V2 Grid Detection


def get_widths(proj):
    pos = []
    width = []
    is_non_zero = False
    for i in range(len(proj)):
        if proj[i] > 0 and not is_non_zero:
            is_non_zero = True
            pos.append(i)
        if proj[i] == 0 and is_non_zero:
            is_non_zero = False
            width.append(i - pos[-1])
    return pos, width
