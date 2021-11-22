import json
import os
import shutil
import sys
import traceback
import zipfile
from io import BytesIO
from pathlib import Path

import cv2
import flask
from authlib.integrations.flask_client import OAuth
from flask import Flask, request, send_file, session, redirect

from Pipeline import process_yeast, saveExtractedRows

CRED = json.load(open('id.json', 'r'))['web']
UPLOAD_FOLDER = "./uploads"

template1 = cv2.imread('upperLeft.png', 0)
template2 = cv2.imread('bottomRight.png', 0)

GOOGLE_CLIENT_ID = CRED['client_id']
GOOGLE_CLIENT_SECRET = CRED['client_secret']

app = Flask("DHY1H-Analyzer")
app.secret_key = "TESTKEY".encode('utf-8') or os.urandom(24)

CONF_URL = 'https://accounts.google.com/.well-known/openid-configuration'
oauth = OAuth(app)
oauth.register(
    client_id=CRED["client_id"],
    auth_uri=CRED["auth_uri"],
    token_uri=CRED["token_uri"],
    auth_provider_x509_cert_url=CRED["auth_provider_x509_cert_url"],
    client_secret=CRED["client_secret"],
    name='google',
    server_metadata_url=CONF_URL,

    client_kwargs={
        'scope': 'openid email profile'
    }
)


@app.route("/")
def home():
    if 'user' in session:
        files = os.listdir('./uploads')
        return flask.render_template('index.html', files=files)
    else:
        redirect_uri = flask.url_for('login', _external=True)
        return oauth.google.authorize_redirect(redirect_uri)


@app.route("/login")
def login():
    token = oauth.google.authorize_access_token()
    user = oauth.google.parse_id_token(token)
    session['user'] = user
    return redirect('/')


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')


@app.route('/createExperiment', methods=['POST'])
def createExperiment():
    if 'user' not in session:
        return redirect("/")
    try:
        if request.method == 'POST':
            experimentName = request.form['experimentName']
            if Path(os.path.join(UPLOAD_FOLDER, experimentName)).exists():
                raise FileExistsError("Experiment already exists")
            Path(os.path.join(UPLOAD_FOLDER, experimentName)).mkdir(parents=True, exist_ok=False)
            output_path = os.path.join(UPLOAD_FOLDER, experimentName)
            excel = request.files['excelFileButton']
            excel.save(os.path.join(output_path, excel.filename))
            for images in request.files.getlist("imagesButton"):
                images.save(os.path.join(output_path, images.filename))
            status, msg = process_yeast(output_path, os.path.join(output_path, excel.filename), template1, template2)
            if status == -1:
                raise Exception(msg)
            else:
                return flask.redirect("/display/" + experimentName)
        else:
            raise Exception("redirecting to home")
    except Exception as err:
        print(err, file=sys.stderr)
        return '''<html><head><meta http-equiv="refresh" content="3;url=/" /> </head><body>{}</body></html>'''.format(
            err)
    return 'ok'


@app.route('/<path:path>', methods=['GET', 'POST'])
def displayExperimentData(path):
    if 'user' not in session:
        return redirect("/")
    if path.startswith("display"):
        try:
            path = path[8:]
            path = os.path.join(UPLOAD_FOLDER, path)
            if not os.path.exists(path):
                raise Exception("Invalid Path :" + path)
            if os.path.isfile(path):
                return flask.send_file(path)
            folderName = path.split('/')[2]
            files = os.listdir(path)
            files.sort()
            return flask.render_template('browser.html', files=files, folderName=folderName)
        except Exception as err:
            return '''<html><head><meta http-equiv="refresh" content="3;url=/" /> </head><body>{}</body></html>'''.format(
                err)
    elif path.startswith('delete'):
        if request.form['password'] == "FuxmanBassLab":
            folder = path.split('/')[-1]
            shutil.rmtree(os.path.join("uploads", folder), ignore_errors=True)
            return flask.redirect('/')
        else:
            return '''<html><head><meta http-equiv="refresh" content="3;url=/" /> </head><body>{}</body></html>'''.format(
                "Incorrect Password")
    elif path.startswith('download'):
        try:
            if path.startswith('download/extract'):
                fileName = request.values['title']
                rows = []
                for key in request.values.keys():
                    if key != 'title':
                        rows.append(int(key))
                if len(rows) == 0:
                    raise Exception("No Rows Selected")
                saveExtractedRows(fileName, rows)
                return flask.redirect('/display' + request.values['title'])
            else:
                if path.split('/')[-1] != '':
                    folder = os.path.join("uploads", path.split('/')[-1])
                    memory_file = BytesIO()
                    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for root, dirs, files in os.walk(folder):
                            for file in files:
                                zipf.write(os.path.join(root, file))
                    memory_file.seek(0)
                    return send_file(memory_file,
                                     attachment_filename=(path.split('/')[-1]) + '.zip',
                                     as_attachment=True)
                else:
                    raise Exception("Invalid path")
        except Exception as err:
            print(traceback.format_exc())
            return '''<html><head><meta http-equiv="refresh" content="3;url=/" /> </head><body>{}</body></html>'''.format(
                err)
    elif path.startswith('templates'):
        return flask.send_file(path)
    else:
        return flask.redirect('/')


if __name__ == "__main__":
    app.run(debug=True)
