import sys
import os
import glob
from disha.utils import extensions,verify_image_name
import pandas as pd

### Usage ###
# Verify files and folder structure before running the algorithm.
# Each Folder must contain Assay Images with png or jpeg format. It should have one Excel file containing transcriptor
# factor information.
# python verify.py <path_to_folder>

def verify_folder():
    exit_flag = False

    if len(sys.argv)!=2 or not os.path.exists(sys.argv[1]):
        print("Folder path missing/invalid \n Usage: python verify.py <path_to_folder>")
        exit()
    root_path = sys.argv[1]


    # Verify Image files extensions and naming format
    print("Verifying images...\n")
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(root_path, '*.{}'.format(ext))))

    if len(images)==0:
        print("No png/jpeg images found")
        exit(-1)

    print("Verifying image names...\n")
    for image in images:
        if not verify_image_name(image):
            print(f"Invalid Image name: {os.path.split(image)[1]}")
            exit_flag = True
    # File names checked
    if exit_flag:
        print("\n\nExpecting file name format as follows:\n\n<experimenter's name>_<experiment name>-<bait#>_<plate_numbers>_<experiment details>",
              "\nExample: AB_HetY1Hcomp-20_5-8_5mm_3AT_Xgal_6d",
              "\nExperimenter's name: AB\nExperiment name: HetY1Hcomp\nBait Number: 20\nPlate Numbers: 5-8\nExperiment details: 5mm_3AT_Xgal_6d")
        exit(-1)
    print("Verifying Excel File...\n")
    excel_file = glob.glob(os.path.join(root_path,'*.xlsx'))

    if len(excel_file)==0:
        print("Corresponding Excel file not found.")
        exit(-1)
    excel_file = excel_file[0]
    print(f"Checking {os.path.split(excel_file)[1]}")
    try:
        datasheet = pd.read_excel(excel_file, engine='openpyxl')
    except Exception as ex:
        print("Failed opening the data file",ex)
        exit(-1)
    expected_columns = ["Coordinate","TF1","TF2"]

    for expected_column in expected_columns:
        if expected_column not in datasheet.columns:
            print(expected_column, " not found")
            exit_flag = True

    if exit_flag:
        print(f"The datafile has columns:{datasheet.columns}\n Expected (case sensitive)columns are:{expected_columns}")
        exit(-1)

    print("Checking datafile contents...")

    coordinates_na = datasheet['Coordinate'].isna()
    if coordinates_na.any():
        print("Empty Cell found in Coordinate Column at\n",datasheet['Coordinate'][coordinates_na].to_string())
        exit_flag=True

    tf1_na = datasheet['TF1'].isna()
    if tf1_na.any():
        print("Empty Cell found in TF1 Column at\n",datasheet['TF1'][tf1_na].to_string())
        exit_flag=True

    tf2_na = datasheet['TF2'].isna()
    if tf2_na.any():
        print("Empty Cell found in TF2 Column at\n",datasheet['TF2'][tf2_na].to_string())
        exit_flag=True

    if exit_flag:
        exit(-1)

    print("Checking Coordinates...")
    regex_matches=datasheet['Coordinate'].str.match(r'[\d]+-[a-zA-Z]{1}\d+')
    if not regex_matches.all():
        print("Invalid coordinates found at\n",datasheet['Coordinate'][~regex_matches].to_string())
        exit(-1)
    print("Folder verified!")



if __name__=="__main__":
    verify_folder()