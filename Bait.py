import cv2


class Bait:
    STATIC_HEADER = '<th>Intensity</th><th>Area</th><th>Image</th>'
    def __init__(self,files:list):
        self.files=files
        self.files.sort()
        self.bait_number=files[0].split("_")[1].split("-")[-1]
        self.days=list(set([fileName.split('.')[0].split("_")[-1][:-1] for fileName in files]))
        self.headers = '<thead><tr>' + '<th colspan="3">Bait Number {}</th>'.format(
            self.bait_number) + '<th colspan="2">TF</th>' + ''.join(['<th colspan="3">Day {}</th>'.format(day) for day in self.days]) \
                       + '</tr>' + '<tr><th>Index</th><th>Activated</th><th>Coordinate</th><th>TF1</th><th>TF2</th>' \
                       + (Bait.STATIC_HEADER * len(self.days)) + '</tr></thead>'
        self.file_map={}
        self.list_array_plates=list(set([fName.split("_")[2] for fName in files]))
        for fileN in files:
            meta_data=fileN.split('.')[0].split("_")
            if meta_data[2] in self.file_map:
                self.file_map[meta_data[2]][meta_data[-1][:-1]]=fileN
            else:
                self.file_map[meta_data[2]]={meta_data[-1][:-1]:fileN}