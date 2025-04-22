from flask import Flask, render_template, request
import numpy as np
from joblib import load
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uuid

app = Flask(__name__)

@app.route("/", methods=["GET",'POST'])
def model_pred():

    request_type_str = request.method
    if request_type_str == "GET":
        return render_template('index.html', href = 'static/base_pic.svg')
    else:
        text = request.form['text']
        random_str = uuid.uuid4().hex
        path = "static/" + random_str + ".svg"
        model = load('model.joblib')
        np_arr = float_to_np(text)
        make_picture('AgesAndHeights.pkl', model, np_arr, path)
        return render_template('index.html', href = path)
        
def make_picture(training_data, model, new_inp,output_file):
  data = pd.read_pickle(training_data)
  ages = data['Age']
  data = data[ages > 0]
  ages = data['Age']
  heights = data['Height']

  x_new = np.array(list(range(19))).reshape((-1,1))
  preds = model.predict(x_new)
  fig = px.scatter(x=ages, y=heights, title='Ages vs Heights', labels= {'x':'Ages (years)',
                                                                        'y':'Heights (inches)'})
  fig.add_trace(go.Scatter(x=x_new.reshape(19),y=preds, mode='lines', name='model'))

  new_preds = model.predict(new_inp)
  fig.add_trace(go.Scatter(x=new_inp.reshape(len(new_inp)),y=new_preds,mode='markers', marker=dict(color='purple',size=20,
                                                                                                  line=dict(color='purple', width = 2))))
  fig.write_image(output_file, width=800, engine='kaleido')
  fig.show()


def float_to_np(float_str):
  def is_float(s):
    try:
      float(s)
      return True
    except:
      return False
  floats = np.array([float(x) for x in float_str.split(',') if is_float(x)])
  return floats.reshape((-1,1))
