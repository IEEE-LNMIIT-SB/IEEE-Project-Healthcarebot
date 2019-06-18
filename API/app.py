from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np

# Your API definition
app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello World!"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        prediction = 1

        return jsonify({'prediction': str(prediction)})

    except:

        return jsonify({'trace': traceback.format_exc()})


if __name__ == '__main__':

    app.run(debug=True)
