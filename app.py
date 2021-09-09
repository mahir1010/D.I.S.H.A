import json
import os
import shutil
import sys
import traceback
import zipfile
from io import BytesIO
from pathlib import Path
import sys
import cv2
import flask
import requests

from saml2 import (
    BINDING_HTTP_POST,
    BINDING_HTTP_REDIRECT,
    entity,
)
from saml2.client import Saml2Client
from saml2.config import Config as Saml2Config

from flask import Flask, request, send_file, session, redirect, url_for

from Pipeline import process_yeast, saveExtractedRows


BYPASS_LOGIN=True


UPLOAD_FOLDER = "./Experiments"

if len(sys.argv)>1 and os.path.isdir(sys.argv[1]):
    UPLOAD_FOLDER=sys.argv[1]

template1 = cv2.imread('upperLeft.png', 0)
template2 = cv2.imread('bottomRight.png', 0)

app = Flask("DHY1H-Analyzer")
app.config['SECRET_KEY'] = "TESTKEY".encode('utf-8') or os.urandom(24)

metadata_URL="https://shib-test.bu.edu/idp/shibboleth"

def saml_client():
    acs_url = url_for(
        "login",
        idp_name="bu.edu",
        _external=True)
    https_acs_url = url_for(
        "login",
        idp_name="bu.edu",
        _external=True,
        _scheme='https')

    #   SAML metadata changes very rarely. On a production system,
    #   this data should be cached as approprate for your production system.
    rv = requests.get(metadata_URL)

    settings = {
        'metadata': {
            'inline': [rv.text],
            },
        'service': {
            'sp': {
                'endpoints': {
                    'assertion_consumer_service': [
                        (acs_url, BINDING_HTTP_REDIRECT),
                        (acs_url, BINDING_HTTP_POST),
                        (https_acs_url, BINDING_HTTP_REDIRECT),
                        (https_acs_url, BINDING_HTTP_POST)
                    ],
                },
                'allow_unsolicited': True,
                'authn_requests_signed': False,
                'logout_requests_signed': True,
                'want_assertions_signed': True,
                'want_response_signed': False,
            },
        },
    }
    spConfig = Saml2Config()
    spConfig.load(settings)
    spConfig.allow_unknown_attributes = True
    saml_client = Saml2Client(config=spConfig)
    return saml_client

@app.route("/")
def home():
    if BYPASS_LOGIN:
        session['user']='bypassed'
    if 'user' in session:
        files = os.listdir(UPLOAD_FOLDER)
        return flask.render_template('index.html', files=files)
    else:
        redirect_uri = flask.url_for('login', _external=True)
        return redirect(redirect_uri)


@app.route("/saml/login")
def login():
    client=saml_client()
    reqid, info = client.prepare_for_authenticate()
    redirect_url = None
    for key, value in info['headers']:
        if key == 'Location':
            redirect_url = value
    response = redirect(redirect_url, code=302)
    response.headers['Cache-Control'] = 'no-cache, no-store'
    response.headers['Pragma'] = 'no-cache'
    return response

@app.route("/saml/sso/", methods=['POST'])
def idp_response():
    client = saml_client()
    authn_response = client.parse_authn_request_response(
        request.form['SAMLResponse'],
        entity.BINDING_HTTP_POST)
    authn_response.get_identity()
    user_info = authn_response.get_subject()
    session['user']=user_info.text
    session['saml_attributes']=authn_response.ava
    return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')


@app.route('/createExperiment', methods=['POST'])
def createExperiment():
    #
    if True:
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
        if type(err)!=FileExistsError:
            shutil.rmtree(os.path.join(UPLOAD_FOLDER, experimentName), ignore_errors=True)
        print(err)
        return '''<html><head><meta http-equiv="refresh" content="10;url=/" /> </head><body>{}</body></html>'''.format(
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
            shutil.rmtree(os.path.join(UPLOAD_FOLDER, folder), ignore_errors=True)
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
                saveExtractedRows(fileName, rows,root_dir=UPLOAD_FOLDER)
                return flask.redirect('/display' + request.values['title'])
            else:
                if path.split('/')[-1] != '':
                    folder = os.path.join(path.split('/')[-1])
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
