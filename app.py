import os
from flask import Flask, request
import numpy as np
import joblib
import sklearn

app = Flask(__name__)
env_config = os.getenv("APP_SETTINGS", "config.DevelopmentConfig")
app.config.from_object(env_config)

def irisPredict(parameters):
  curr_dir = os.path.dirname(__file__)
  file_path = curr_dir + '/nn.pkl'
  file_path_2 = './nn.pkl'
  print('getcwd:      ', os.getcwd())
  print('__file__:    ', __file__)
  print('abspath:     ', os.path.abspath(__file__))
  print('abs dirname: ', os.path.dirname(os.path.abspath(__file__)))
  print('cur_dir: ', curr_dir)
  print('file_path: ', file_path)
  print('fp_2: ', file_path_2)
  model = joblib.load(file_path)
  print('model: ', model)
  params = parameters.reshape(1, -1)
  pred = model.predict(params)
  return file_path

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

@app.route('/')
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

if __name__ == '__app__':
  app.run(debug=True)