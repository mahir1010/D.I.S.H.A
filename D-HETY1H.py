from Pipeline import *
import sys
import os
import logging
import glob
from datetime import datetime
import traceback

extensions=['[jJ][pP][gG]','[pP][nN][gG]']

now=datetime.now()

date = now.strftime("%m-%d-%Y-%H-%M-%S")
if len(sys.argv)!=2:
    print("usage: D-HY1H.py <root Experiment Folder Path>")
    exit(-1)

rootPath=sys.argv[1]
if not os.path.isdir(rootPath):
    print("Invalid path: ",rootPath)
    exit(-1)

logging.basicConfig(filename=date+'.log',level=logging.INFO)
contents=os.listdir(rootPath)
experimentDirectories=[]
template1 = cv2.imread('upperLeft.png', 0)
template2 = cv2.imread('bottomRight.png', 0)

for potentialDir in contents:
    potentialPath=os.path.join(rootPath,potentialDir)
    if not os.path.isdir(potentialPath):
        continue
    imgs=[]
    for ext in extensions:
        imgs.extend(glob.glob(os.path.join(potentialPath,'*.{}'.format(ext))))
    excel=glob.glob(os.path.join(potentialPath,'*.xlsx'))
    if len(imgs)==0 or len(excel)==0:
        continue
    if os.path.isdir(os.path.join(potentialPath,'output')):
        continue
    experimentDirectories.append(potentialPath)
if len(experimentDirectories)==0:
    logging.error("Can not find any new directory")
    exit(-1)
logging.info("Found Following directories:\n"+"\n".join(experimentDirectories))
for dir in experimentDirectories:
    logging.info("Processing :"+dir)
    excel_path=glob.glob(os.path.join(dir,'*.xlsx'))[0]
    try:
        process_yeast(dir,excel_path,template1,template2)
    except Exception as e:
        logging.error("Error while Processing {}: {} \n\n{}".format(dir,str(e),traceback.format_exc()))




