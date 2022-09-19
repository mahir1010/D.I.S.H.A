# Heterodimer-Y1H-Analysis

An automated tool to detect and analyze yeast-based assays to learn more about transcriptional regulation in immune
responses and cancer. We are working for
[Anna Berenson](https://www.bu.edu/mcbb/profile/anna-berenson/) and [Prof. Fuxman Bass](https://www.fuxmanlab.com/).

The detailed methodology is explained more in
the [DY1H-Slide Deck.pdf](https://github.com/mahir1010/Heterodimer-Y1H-Analysis/blob/3f4fb20688b8a6ce9ea7bb8650551d4ccdfe8e71/DY1H-Slide%20Deck.pdf)
. Please download it first to read it with proper formatting.

## Instructions

To launch the flask application, open a terminal at the root of this directory and copy paste this command :

``` FLASK_APP=app.py flask run ```

After that, on your localhost adress, you will be prompted to specify your excel files containing your input images
coordinates with respect to the grid cell, and the images you want to process. Once it is done, click the submit button.

The output of the flask application will contain the extracted quads, their mask and pre_mask features, and their
corresponding pdfs and html files.

## Code and repository description

The main python file is
called [Pipeline.py](https://github.com/mahir1010/Heterodimer-Y1H-Analysis/blob/main/Pipeline.py), containing the source
code pre-processing the images, generating a grid cell and extracting the intensity and area of each quad.

The [app.py](https://github.com/mahir1010/Heterodimer-Y1H-Analysis/blob/main/app.py) python file contains the flask
application.

The [data](https://github.com/mahir1010/Heterodimer-Y1H-Analysis/blob/main/data/) folder contains the same plate images
at a different time, with their caracteristics and coordinate location
in [HetY1H Pilot TF coordinates.xlsx](https://github.com/mahir1010/Heterodimer-Y1H-Analysis/blob/main/data/HetY1H%20Pilot%20TF%20coordinates.xlsx)
.

The [templates](https://github.com/mahir1010/Heterodimer-Y1H-Analysis/blob/main/templates/) folder contains html files
for the flask application.

## Results

![html_result](https://github.com/mahir1010/Heterodimer-Y1H-Analysis/blob/main/res_16_5.png "results")
With the above image, we can see that for each image analyzed, for each coordinate, we have its tf1 and tf2 values, its
raw intensity, its image, and references quad information.

## Discussion

Our intensity and area calculation are pretty accurate, but our detection threshold needs some more fine-tuning. Right
now, we map our score to an exponential distribution to match Anna's score, but we are currently exploring other
approaches, like semi-supervised learning.
