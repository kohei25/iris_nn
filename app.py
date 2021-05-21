from flask import Flask, json, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)
app.config['DEBUG'] = True

def irisPredict(parameters):
  model = joblib.load('./nn.pkl')
  params = parameters.reshape(1, -1)
  pred = model.predict(params)
  return pred

def getName(label):
  print(label)
  if label == 0:
    return 'Iris Setosa'
  elif label == 1:
    return 'Iris Versicolor'
  elif label == 2:
    return 'Iris Virginica'
  else:
    return 'Error'

@app.route('/', methods=['GET'])
def home():
  return '<h1>hello</h1>'

@app.route('/api/v1/predict', methods=['GET'])
def predict():
  query_parameters = request.args
  if query_parameters.get('sl') and query_parameters.get('sw') and query_parameters.get('pl') and query_parameters.get('pw'):
    sepal_length = float(query_parameters.get('sl'))
    sepal_width = float(query_parameters.get('sw'))
    petal_length = float(query_parameters.get('pl'))
    petal_width = float(query_parameters.get('sw'))
    x = np.array([sepal_length, sepal_width, petal_length, petal_width])
    pred = irisPredict(x)
    iris_name = getName(pred)
    return iris_name
  else:
    return 'error'
