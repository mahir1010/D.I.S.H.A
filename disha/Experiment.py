import bz2
import math
import os
import pickle
import traceback

import numpy as np
import pandas as pd

from disha.Image import Image
from disha.utils import extract_plate_name


def convert_index_to_plate(coords):
    index = (coords.split('-')[0]).strip()
    base = math.ceil(int(index) / 4) * 4
    return f'{base - 3}-{base}'


def _generate_normalization_fn(normalization_type, intensities: np.ndarray, areas: np.ndarray, ignore_zero_values):
    fn_intensity = lambda x: x
    fn_area = lambda x: x
    if normalization_type == "Percent":
        min_val_intensity = intensities.min()
        max_val_intensity = intensities.max()
        min_val_area = areas.min()
        max_val_area = areas.max()
        fn_intensity = lambda x: round((x - min_val_intensity) / (max_val_intensity - min_val_intensity) * 100,
                                       2) if x != 0 or not ignore_zero_values else 0
        fn_area = lambda x: round((x - min_val_area) / (max_val_area - min_val_area) * 100,
                                  2) if x != 0 or not ignore_zero_values else 0

    elif normalization_type == "Z-Score":
        mean_intensity = np.mean(intensities)
        sd_intensity = np.std(intensities)
        mean_area = np.mean(areas)
        sd_area = np.std(areas)
        fn_intensity = lambda x: round((x - mean_intensity) / sd_intensity,
                                       2) if x != 0 or not ignore_zero_values else 0
        fn_area = lambda x: round((x - mean_area) / sd_area, 2) if x != 0 or not ignore_zero_values else 0
    elif normalization_type == "Min-Offset":
        min_intensity = intensities[intensities > 0].min()
        min_area = areas[areas > 0].min()
        fn_intensity = lambda x: round(x - min_intensity,2) if x != 0 else 0
        fn_area = lambda x: round(x - min_area,2) if x != 0 else 0
    return fn_intensity, fn_area


class Experiment:

    def __init__(self, bait, images, datasheet, base_path, initialize=True):
        self.bait = bait
        self.base_path = base_path
        self.output_path = os.path.join(base_path, 'output')
        if initialize:
            self.datasheet = pd.read_excel(datasheet, engine='openpyxl')
            self.datasheet['plate'] = self.datasheet['Coordinate'].apply(lambda x: convert_index_to_plate(x))
            self.images = [
                Image(image, self.output_path,
                      self.datasheet[self.datasheet['plate'] == extract_plate_name(os.path.basename(image))]) for
                image in images]
        else:
            self.datasheet = datasheet
            self.images = images
        self.plate_map = {}
        for image in self.images:
            if image.plate_base_number in self.plate_map:
                self.plate_map[image.plate_base_number].append(image)
            else:
                self.plate_map[image.plate_base_number] = [image]

    def apply_funct(self, funct, params=None):
        for image in self.images:
            if image.exception_occurred:
                continue
            try:
                if params is None:
                    yield image, funct(image)
                else:
                    yield image, funct(image, params)
            except Exception as e:
                image.exception_occurred = True
                image.exception_reason = str(e)
                print(f"Error while processing :{image}")
                traceback.print_exc()

    def export_excel_sheet(self):
        for image in self.images:
            if image.exception_occurred:
                continue
            dataframe = image.dataframe
            name = image.name
            excel_columns = list(dataframe.columns)
            excel_columns.remove('Image')
            dataframe[excel_columns].to_excel(os.path.join(self.output_path, name.split('.')[0]) + ".xlsx", index=False)

    def generate_evaluation_table(self, empty_coordinates=[]):
        assert type(empty_coordinates) == list
        while (len(empty_coordinates)) < len(self.images):
            empty_coordinates.append('')
        for image, empty_coordinate in zip(self.images, empty_coordinates):
            image.generate_evaluation_table(empty_coordinate)

    def normalize_data(self, normalization_type="Percent", intensity_column=True, area_column=True,
                       ignore_zero_values=True, per_plate=False):
        if not intensity_column and not area_column:
            return
        intensities = []
        areas = []
        if not per_plate:
            for image in self.images:
                if image.exception_occurred:
                    continue
                image.dataframe.reset_index(drop=True, inplace=True)
                image.raw_dataframe.reset_index(drop=True, inplace=True)
                intensities.extend(image.raw_dataframe['Intensity'].tolist())
                areas.extend(image.raw_dataframe['Area'].tolist())
            intensities = np.array(intensities)
            areas = np.array(areas)
            if ignore_zero_values:
                intensities = intensities[intensities != 0]
                areas = areas[areas != 0]
            fn_intensity, fn_area = _generate_normalization_fn(normalization_type, intensities, areas,
                                                               ignore_zero_values)
        for image in self.images:
            if image.exception_occurred:
                continue
            if per_plate:
                image.dataframe.reset_index(drop=True, inplace=True)
                image.raw_dataframe.reset_index(drop=True, inplace=True)
                intensities = np.array(image.raw_dataframe['Intensity'].tolist())
                areas = np.array(image.raw_dataframe['Area'].tolist())
                if ignore_zero_values:
                    intensities = intensities[intensities != 0]
                    areas = areas[areas != 0]
                if len(intensities)==0 or len(areas)==0:
                    continue
                fn_intensity, fn_area = _generate_normalization_fn(normalization_type, intensities, areas,
                                                                   ignore_zero_values)
            if intensity_column:
                image.dataframe['Intensity'] = image.raw_dataframe['Intensity'].apply(fn_intensity)
            if area_column:
                image.dataframe['Area'] = image.raw_dataframe['Area'].apply(fn_area)
            image.generate_evaluation_table(image.reference_coords)

    def generate_heatmap(self, files=[]):
        assert type(files) == list
        target_images = []
        for image in self.images:
            if image.name in files:
                target_images.append(image)
        transcription_factor_1 = []
        transcription_factor_2 = []
        for image in target_images:
            transcription_factor_1.extend(image.raw_dataframe['TF1'].unique().tolist())
            transcription_factor_2.extend(image.raw_dataframe['TF2'].unique().tolist())
        transcription_factor_1 = list(set(transcription_factor_1))
        transcription_factor_1 = [tf for tf in transcription_factor_1 if type(tf) == str]
        transcription_factor_2 = list(set(transcription_factor_2))
        transcription_factor_2 = [tf for tf in transcription_factor_2 if type(tf) == str]
        heatmap_intensity = np.zeros((len(transcription_factor_1), len(transcription_factor_2)))
        heatmap_area = np.zeros((len(transcription_factor_1), len(transcription_factor_2)))
        for row in range(len(transcription_factor_2)):
            for column in range(len(transcription_factor_1)):
                intensity = 0
                area = 0
                for image in target_images:
                    data = image.get_details_tf(transcription_factor_1[column], transcription_factor_2[row])
                    intensity += data[0]
                    area += data[1]
                heatmap_intensity[column][row] = round(intensity / len(target_images), 3)
                heatmap_area[column][row] = round(area / len(target_images), 3)
        return transcription_factor_1, transcription_factor_2, heatmap_intensity, heatmap_area


def export_experiment(experiment, single_file=True):
    if single_file:
        pickle.dump(experiment,
                    bz2.BZ2File(os.path.join(experiment.base_path, f'bait_{experiment.bait}_results.dhy1h'), 'wb'))
    else:
        for plate in experiment.plate_map.keys():
            imgs = experiment.plate_map[plate]
            sub_experiment = Experiment(experiment.bait, imgs, experiment.datasheet, experiment.base_path, False)
            pickle.dump(sub_experiment,
                        bz2.BZ2File(
                            os.path.join(sub_experiment.base_path, f'bait_{sub_experiment.bait}_{plate}_results.dhy1h'),
                            'wb'))
            del sub_experiment
