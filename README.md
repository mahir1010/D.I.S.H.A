# Heterodimer-Y1H-Analysis
An automated tool to detect and analyze yeast-based assays to learn more about transcriptional regulation in immune responses and cancer. We are working for
[Anna Berenson](https://www.bu.edu/mcbb/profile/anna-berenson/) and [Prof. Fuxman Bass](https://www.fuxmanlab.com/).

The detailed methodology is explained more in the [DY1H-Slide Deck.pdf](https://github.com/mahir1010/Heterodimer-Y1H-Analysis/blob/3f4fb20688b8a6ce9ea7bb8650551d4ccdfe8e71/DY1H-Slide%20Deck.pdf). Please download it first to read it with proper formatting.

## Instructions
To launch the flask application, open a terminal at the root of this directory and copy paste this command : 

``` FLASK_APP=app.py flask run ```

## Code and repository description
The main python file is called [Pipeline.py](https://github.com/mahir1010/Heterodimer-Y1H-Analysis/blob/main/Pipeline.py), containing the source code pre-processing the images, generating a grid cell and extracting the intensity and area of each quad.

The [app.py](https://github.com/mahir1010/Heterodimer-Y1H-Analysis/blob/main/app.py) python file  contains the flask application.

The [data](https://github.com/mahir1010/Heterodimer-Y1H-Analysis/blob/main/data/) folder contains the same plate images at a different time, with their caracteristics and coordinate location in [HetY1H Pilot TF coordinates.xlsx](https://github.com/mahir1010/Heterodimer-Y1H-Analysis/blob/main/data/HetY1H Pilot TF coordinates.xlsx).

The [templates](https://github.com/mahir1010/Heterodimer-Y1H-Analysis/blob/main/templates) folder contains html files for the flask application.

## Results

## Discussion
