from flask import Flask, request
from model import predictRisk
from pandas import DataFrame

app = Flask(__name__)

@app.route("/", methods=['POST'])
def risk():
    data = request.get_json()
    predictedRisk = predictRisk(DataFrame(data))
    return predictedRisk