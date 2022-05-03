import os

from flask import Flask, request, jsonify, render_template
from main import datasetpreparation, severityanalysis,datasetpreparationRAW
from PIL import Image
from flask import Flask, flash, request, redirect, render_template, jsonify,url_for, send_file
from werkzeug.utils import secure_filename
import pandas as pd
DatasetCCCM, DatasetEducation, DatasetHealth, DatasetNutrition, DatasetProtection, DatasetNFI, DatasetShelter, DatasetWASH, DatasetFS, DatasetERL, DatasetIntersector = datasetpreparation(
    'datasetsyria2.csv')
DatasetCCCM1, DatasetEducation1, DatasetHealth1, DatasetNutrition1, DatasetProtection1, DatasetNFI1, DatasetShelter1, DatasetWASH1, DatasetFS1, DatasetERL1, DatasetIntersector1 = datasetpreparationRAW(
    'datasetsyria2.csv')

app = Flask(__name__)

app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
# Get current path
path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'static/uploads')

# Make directory if uploads is not exists
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/DatasetCCCM', methods=['POST'])
def proceed1():

#results_final=severityanalysis(DatasetCCCM)
    #print(results_final)
    return DatasetCCCM

@app.route('/DatasetEducation', methods=['POST'])
def proceed2():

    return DatasetEducation

@app.route('/DatasetHealth', methods=['POST'])
def proceed3():
    results_final = severityanalysis(pd.DataFrame(DatasetHealth))
#results_final=severityanalysis(DatasetCCCM)
    #print(results_final)
    return DatasetHealth,results_final

@app.route('/DatasetNutrition', methods=['POST'])
def proceed4():

    return DatasetNutrition

@app.route('/DatasetProtection', methods=['POST'])
def proceed5():

    return DatasetProtection

@app.route('/DatasetNFI', methods=['POST'])
def proceed6():

    return DatasetNFI

@app.route('/DatasetShelter', methods=['POST'])
def proceed7():

    return DatasetShelter

@app.route('/DatasetWASH', methods=['POST'])
def proceed8():

    return DatasetWASH

@app.route('/DatasetFS', methods=['POST'])
def proceed9():

    return DatasetFS

@app.route('/DatasetIntersector', methods=['POST'])
def proceed10():

    return DatasetIntersector

@app.route('/DatasetERL', methods=['POST'])
def proceed11():

    return DatasetERL

@app.route('/MLDatasetCCCM', methods=['POST'])
def proceedml1():
    results_final = severityanalysis(pd.DataFrame(DatasetCCCM1))
#results_final=severityanalysis(DatasetCCCM)
    #print(results_final)
    return jsonify(results_final)

@app.route('/MLDatasetEducation', methods=['POST'])
def proceedml2():
    results_final = severityanalysis(pd.DataFrame(DatasetEducation1))
#results_final=severityanalysis(DatasetCCCM)
    #print(results_final)
    return jsonify(results_final)

@app.route('/MLDatasetHealth', methods=['POST'])
def proceedml3():
    results_final = severityanalysis(pd.DataFrame(DatasetHealth1))
#results_final=severityanalysis(DatasetCCCM)
    #print(results_final)
    return jsonify(results_final)

@app.route('/MLDatasetNutrition', methods=['POST'])
def proceedml4():
    results_final=severityanalysis(pd.DataFrame(DatasetNutrition1))
#results_final=severityanalysis(DatasetCCCM)
    #print(results_final)
    return jsonify(results_final)

@app.route('/MLDatasetProtection', methods=['POST'])
def proceedml5():
    results_final=severityanalysis(pd.DataFrame(DatasetProtection1))
#results_final=severityanalysis(DatasetCCCM)
    #print(results_final)
    return jsonify(results_final)

@app.route('/MLDatasetNFI', methods=['POST'])
def proceedml6():
    results_final=severityanalysis(pd.DataFrame(DatasetNFI1))
#results_final=severityanalysis(DatasetCCCM)
    #print(results_final)
    return jsonify(results_final)

@app.route('/MLDatasetShelter', methods=['POST'])
def proceedml7():
    results_final=severityanalysis(pd.DataFrame(DatasetShelter1))
#results_final=severityanalysis(DatasetCCCM)
    #print(results_final)
    return jsonify(results_final)

@app.route('/MLDatasetWASH', methods=['POST'])
def proceedml8():
    results_final=severityanalysis(pd.DataFrame(DatasetWASH1))
#results_final=severityanalysis(DatasetCCCM)
    #print(results_final)
    return jsonify(results_final)

@app.route('/MLDatasetFS', methods=['POST'])
def proceedml9():
    results_final=severityanalysis(pd.DataFrame(DatasetFS1))
#results_final=severityanalysis(DatasetCCCM)
    #print(results_final)
    return jsonify(results_final)

@app.route('/MLDatasetIntersector', methods=['POST'])
def proceedml10():
    results_final=severityanalysis(pd.DataFrame(DatasetIntersector1))
#results_final=severityanalysis(DatasetCCCM)
    #print(results_final)
    return jsonify(results_final)

@app.route('/MLDatasetERL', methods=['POST'])
def proceedml11():
    results_final=severityanalysis(pd.DataFrame(DatasetERL1))
#results_final=severityanalysis(DatasetCCCM)
    #print(results_final)
    return jsonify(results_final)

@app.route('/')
def index2():
    return render_template('index.html')
from waitress import serve
from waitress import serve
if __name__ == "__main__":
    # app.run("0.0.0.0",port=5000, debug=True)
    #app.run(debug=True)
    # app.run(host='0.0.0.0', port=port) # <---- REMOVE THIS
    # serve your flask app with waitress, instead of running it directly.
    serve(app, host='0.0.0.0',port=5000)  # <---- ADD THIS

