import logging
import os
import traceback
from pathlib import Path

from PIL import Image as PIL_Image
from PIL import ImageDraw

from disha.utils import *


class Image:
    def __init__(self, file_path, output_path, dataframe):
        self.dataframe = dataframe.copy()
        self.plate_base_number = int(dataframe['plate'].iloc[0].split('-')[0])
        self.file_path = file_path
        self.name = Path(file_path).stem
        self.image = np.array(PIL_Image.open(file_path))
        if self.image is None:
            logging.error("Error while Processing {}:\n\n{}".format(file_path, traceback.format_exc()))
        self.output_colonies_path = os.path.join(output_path, self.name.split('.')[0])
        Path(self.output_colonies_path).mkdir(parents=True, exist_ok=True)
        self.display_dataframe = None
        self.raw_dataframe = None
        self.intensity_map = None
        self.exception_occurred = False
        self.exception_reason = None
        self.reference_coords = ''

    def getIndex(self, coords):
        index = coords.split("-")
        index[0] = int(index[0])
        yIndex = row2IndexMap[index[1][0]]
        xIndex = int(index[1][1:]) - 1
        colonies = []

        if index[0] % 4 == 1:
            xIndex = 4 * xIndex
            yIndex = 4 * yIndex
        elif index[0] % 4 == 2:
            xIndex = 4 * xIndex + 2
            yIndex = 4 * yIndex
        elif index[0] % 4 == 3:
            xIndex = 4 * xIndex
            yIndex = 4 * yIndex + 2
        elif index[0] % 4 == 0:
            xIndex = 4 * xIndex + 2
            yIndex = 4 * yIndex + 2

        for i in range(2):
            for j in range(2):
                colonies.append(
                    [[self.rows[xIndex + i] - self.rows[xIndex], self.rows[xIndex + 1 + i] - self.rows[xIndex]],
                     [self.cols[yIndex + j] - self.cols[yIndex], self.cols[yIndex + j + 1] - self.cols[yIndex]]])
        return [[self.rows[xIndex], self.rows[xIndex + 2]], [self.cols[yIndex], self.cols[yIndex + 2]]], colonies

    # Returns TF1 TF2 Intensity Area
    def getDetails(self, coords):
        return self.dataframe[self.dataframe['Coordinate'] == coords][['TF1', 'TF2', 'Intensity', 'Area']].values[0]

    def set_grid(self, rows, cols):
        self.rows = rows
        self.cols = cols

    def export_grid_image(self):
        image = PIL_Image.fromarray(self.image.copy())
        draw = ImageDraw.Draw(image)
        for i, p in enumerate(self.rows):
            if i % 4 == 0:
                draw.line([(p, 0), (p, image.size[1] - 1)], (0, 0, 255), width=3)
            elif i % 2 == 0:
                draw.line([(p, 0), (p, image.size[1] - 1)], (0, 0, 153), width=3)
            else:
                draw.line([(p, 0), (p, image.size[1] - 1)], (0, 0, 0), width=3)
        for i, p in enumerate(self.cols):
            if i % 4 == 0:
                draw.line([(0, p), (image.size[0] - 1, p)], (0, 0, 255), width=3)
            elif i % 2 == 0:
                draw.line([(0, p), (image.size[0] - 1, p)], (0, 0, 153), width=3)
            else:
                draw.line([(0, p), (image.size[0] - 1, p)], (0, 0, 0), width=3)
        return np.array(image)

    def get_coordinate(self, x, y, transpose=False):
        rows, cols = [self.rows, self.cols] if not transpose else [self.cols, self.rows]
        row_index = col_index = 0
        if x < rows[0] or x > rows[-1] or y < cols[0] or y > cols[-1]:
            return ''
        for i in range(0, len(rows), 4):
            if x < rows[i + 4]:
                row_index = i
                break
        for i in range(0, len(cols), 4):
            if y < cols[i + 4]:
                col_index = i
                break
        coords = f"{Index2rowMap[math.floor(col_index / 4)]}{math.floor(row_index / 4) + 1:02d}"
        if x < rows[row_index + 2] and y < cols[col_index + 2]:
            quad_id = 0
        elif x < rows[row_index + 4] and y < cols[col_index + 2]:
            quad_id = 1
        elif x < rows[row_index + 2] and y < cols[col_index + 4]:
            quad_id = 2
        else:
            quad_id = 3
        return f"{self.plate_base_number + quad_id}-{coords}"

    def __str__(self):
        return self.name

    def generate_evaluation_table(self, empty_coordinate):
        if self.exception_occurred:
            return
        self.reference_coords = empty_coordinate
        dataframe = self.dataframe
        dataframe.reset_index(drop=True, inplace=True)
        if empty_coordinate == '':
            empty_locations = dataframe.loc[(dataframe['TF2'] == "empty") & (dataframe['TF1'] == "empty")]
            empty_locations.sort_values(by=['Intensity'],inplace=True,ascending=False)
        else:
            empty_locations = dataframe.loc[dataframe['Coordinate'] == empty_coordinate]
        empty_empty_rs = (empty_locations['Intensity'] * empty_locations['Area']).mean()

        empty = empty_locations.iloc[0]
        # dataframe['RS']=dataframe.apply(lambda row: (row['Intensity']*row['Area'])-empty_empty_rs if row['Intensity']!=0 else 0)
        target_df = dataframe.loc[(dataframe['TF2'] != "empty") & (dataframe['TF1'] != "empty")].copy(deep=True)
        target_df['Coop_Index'] = 0
        target_df['Antagonism_1'] = 0
        target_df['Antagonism_2'] = 0
        target_df['tf1empty_intensity'] = target_df['Intensity']
        target_df['tf1empty_area'] = target_df['Area']
        target_df['tf1empty_image'] = target_df['Image']
        target_df['tf2empty_intensity'] = target_df['Intensity']
        target_df['tf2empty_area'] = target_df['Area']
        target_df['tf2empty_image'] = target_df['Image']
        target_df['empty_empty_intensity'] = target_df['Intensity']
        target_df['empty_empty_area'] = target_df['Area']
        target_df['empty_empty_image'] = target_df['Image']
        target_df['empty_empty_image'] = target_df['empty_empty_image'].apply(lambda x: empty['Image'])
        target_df['empty_empty_intensity'] = target_df['empty_empty_intensity'].apply(lambda x: empty['Intensity'])
        target_df['empty_empty_area'] = target_df['empty_empty_area'].apply(lambda x: empty['Area'])
        target_df['empty_empty_area'] = target_df['empty_empty_area'].apply(lambda x: empty['Area'])
        for index, row in target_df.iterrows():
            if type(row['TF1']) != str or type(row['TF2']) != str:
                print("Potential Error:", row)
                continue
            if len(dataframe.loc[(dataframe['TF2'] == "empty") & (dataframe['TF1'] == row['TF1'])]) == 0:
                print("Potential Error:", row)
                continue
            tf1_tf2_rs = (row['Intensity'] * row['Area']) - empty_empty_rs if row['Intensity']!=0 else 0
            temp = dataframe.loc[(dataframe['TF2'] == "empty") & (dataframe['TF1'] == row['TF1'])].iloc[0]
            tf1_empty_rs = (temp['Intensity'] * temp['Area']) - empty_empty_rs
            target_df.loc[index, 'tf1empty_intensity'] = temp['Intensity']
            target_df.loc[index, 'tf1empty_area'] = temp['Area']
            target_df.loc[index, 'tf1empty_image'] = temp['Image']
            if len(dataframe.loc[(dataframe['TF1'] == "empty") & (dataframe['TF2'] == row['TF2'])]) == 0:
                print("Potential Error:", row)
                continue
            temp = dataframe.loc[(dataframe['TF1'] == "empty") & (dataframe['TF2'] == row['TF2'])].iloc[0]
            tf2_empty_rs = (temp['Intensity'] * temp['Area']) - empty_empty_rs
            target_df.loc[index, 'tf2empty_intensity'] = temp['Intensity']
            target_df.loc[index, 'tf2empty_area'] = temp['Area']
            target_df.loc[index, 'tf2empty_image'] = temp['Image']
            target_df.loc[index, 'Coop_Index'] = round(tf1_tf2_rs - tf1_empty_rs - tf2_empty_rs, 2)
            target_df.loc[index, 'Antagonism_1'] = round(tf1_empty_rs - tf1_tf2_rs, 2)
            target_df.loc[index, 'Antagonism_2'] = round(tf2_empty_rs - tf1_tf2_rs, 2)
        if self.display_dataframe is None or 'Activated' not in self.display_dataframe:
            target_df.insert(0, 'Activated', [False for i in range(len(target_df.index))])
        else:
            target_df.insert(0, 'Activated', self.display_dataframe['Activated'].tolist())
        # target_df['Coop_Index'] = (2 * ((target_df['Coop_Index'] - target_df['Coop_Index'].min()) / (
        #             target_df['Coop_Index'].max() - target_df['Coop_Index'].min())) - 1).round(2)
        # target_df['Antagonism_1'] = (2 * ((target_df['Antagonism_1'] - target_df['Antagonism_1'].min()) / (
        #         target_df['Antagonism_1'].max() - target_df['Antagonism_1'].min())) - 1).round(2)
        # target_df['Antagonism_2'] = (2 * ((target_df['Antagonism_2'] - target_df['Antagonism_2'].min()) / (
        #         target_df['Antagonism_2'].max() - target_df['Antagonism_2'].min())) - 1).round()

        self.display_dataframe = target_df
        self.display_dataframe.reset_index(drop=True, inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)

    def get_details_tf(self, tf1, tf2):
        data = self.dataframe.loc[
            (self.dataframe['TF1'] == tf1) & (self.dataframe['TF2'] == tf2), ['Intensity', 'Area']].values
        if len(data) > 0:
            return data[0]
        else:
            return [0, 0]
