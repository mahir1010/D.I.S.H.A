import os
from pathlib import Path

from flask import Flask,request
import flask
from Pipeline import process_yeast
import cv2

app = Flask(__name__)
UPLOAD_FOLDER="./uploads"

template1 = cv2.imread('upperLeft.png', 0)
template2 = cv2.imread('bottomRight.png', 0)
@app.route("/")
def home():
    files=os.listdir('./uploads')
    return flask.render_template('index.html',files=files)

@app.route('/createExperiment',methods = ['POST'])
def createExperiment():
    try:
        if request.method == 'POST':
            experimentName=request.form['experimentName']
            if Path(os.path.join(UPLOAD_FOLDER , experimentName)).exists():
                raise FileExistsError("Experiment already exists")
            Path(os.path.join(UPLOAD_FOLDER , experimentName)).mkdir(parents=True, exist_ok=False)
            output_path=os.path.join(UPLOAD_FOLDER,experimentName)
            excel=request.files['excelFileButton']
            excel.save(os.path.join(output_path, excel.filename))
            for images in request.files.getlist("imagesButton"):
                images.save(os.path.join(output_path,images.filename))
            status,msg=process_yeast(output_path,os.path.join(output_path, excel.filename),template1,template2)
            if status==-1:
                raise Exception(msg)
            else:
                return flask.redirect("/display/"+experimentName)
        else:
            raise Exception("redirecting to home")
    except Exception as err:
        return '''<html><head><meta http-equiv="refresh" content="3;url=/" /> </head><body>{}</body></html>'''.format(err)
    return 'ok'

@app.route('/<path:path>',methods=['GET'])
def displayExperimentData(path):
    if path.startswith("display"):
        path=path[8:]
    else:
        return flask.redirect('/')
    try:
        path =os.path.join(UPLOAD_FOLDER,path)
        if not os.path.exists(path):
            raise Exception("Invalid Path :"+path)
        if os.path.isfile(path):
            return flask.send_file(path)
        files = os.listdir(path)
        files.sort()
        return flask.render_template('browser.html', files=files)
    except Exception as err:
        return '''<html><head><meta http-equiv="refresh" content="3;url=/" /> </head><body>{}</body></html>'''.format(
            err)
if __name__ == "__main__":
    app.run(debug=True)
