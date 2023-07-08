import glob
import sys
from datetime import datetime

from disha.Pipeline import *
from disha.utils import extract_bait_number,extensions


def disha_analyzer():
    now = datetime.now()

    date = now.strftime("%m-%d-%Y-%H-%M-%S")
    if len(sys.argv) != 4:
        print("usage: D-HY1H.py <root Experiment Folder Path> <path_model_weights> <batch_size>")
        exit(-1)

    rootPath = sys.argv[1]
    model_weights = sys.argv[2]
    batch_size = int(sys.argv[3])
    if not os.path.isdir(rootPath):
        print("Invalid path: ", rootPath)
        exit(-1)
    if not os.path.isdir(model_weights):
        print("Invalid path: ", model_weights)
        exit(-1)
    logging.basicConfig(filename=date + '.log', level=logging.INFO)
    contents = os.listdir(rootPath)
    experimentDirectories = {}

    for potentialDir in contents:
        potentialPath = os.path.join(rootPath, potentialDir)
        if not os.path.isdir(potentialPath):
            continue
        imgs = []
        for ext in extensions:
            imgs.extend(glob.glob(os.path.join(potentialPath, '*.{}'.format(ext))))
        excel = glob.glob(os.path.join(potentialPath, '*.xlsx'))
        if len(imgs) == 0 or len(excel) == 0:
            continue
        if os.path.isdir(os.path.join(potentialPath, 'output')):
            continue
        experimentDirectories[potentialPath] = {'images': imgs, 'data': excel[0]}
    if len(experimentDirectories) == 0:
        logging.error("Can not find any new directory")
        exit(-1)
    logging.info("Found Following directories:\n" + "\n".join(experimentDirectories))
    experiments = {}

    for dir in experimentDirectories:
        logging.info("Processing :" + dir)
        image_map = {}
        for image in experimentDirectories[dir]['images']:
            bait = extract_bait_number(os.path.basename(image))
            if bait not in image_map:
                image_map[bait] = []
            image_map[bait].append(image)
        for bait in image_map:
            try:
                experiments[dir + bait] = Experiment(bait, image_map[bait], experimentDirectories[dir]['data'], dir)
            except Exception as e:
                logging.error(
                    "creating experiment failed: Likely reason, plate information not found in the corresponding data file")
                logging.error(str(e))

    experiments = list(experiments.values())

    for experiment in experiments:
        process_experiment(experiment, model_weights, batch_size=batch_size)

if __name__=="__main__":
    disha_analyzer()