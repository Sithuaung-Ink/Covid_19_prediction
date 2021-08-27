import os
import numpy as np
import flask
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras


app=Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,8)
    loaded_model = keras.models.load_model("covid_prediction_model_2.h5")
    result = loaded_model.predict(to_predict)
    return result

@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        if result > 0.05:
            prediction='Positive'
        else:
            prediction='Negative'
        return render_template("result.html",prediction=prediction)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
