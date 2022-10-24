from flask import Flask, Flask,render_template,url_for,request
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import pickle
import warnings
warnings.filterwarnings("ignore")
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('Index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    model = pickle.load(open("model.pkl", "rb"))
    if request.method=='POST':
        d= request.form.to_dict()
        input_array =[]
        for col in model.feature_names_:
            value= d.get(col)
            input_array.append(value)
    y_pred = model.predict(input_array)
    return render_template('Predict.html')

if __name__ == '__main__':
    app.run(debug=True)