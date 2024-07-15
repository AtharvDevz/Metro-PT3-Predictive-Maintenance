# -*- coding: utf-8 -*-
import uvicorn
from fastapi import FastAPI
from sensorData import sensor_data
import numpy as np
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

# Making an API
app = FastAPI()
pickle_in = open(r"Models\LGBMModel.sav","rb")
model=pickle.load(pickle_in)


@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.post('/predict')
def predict_failure(data:sensor_data):
    print("data + ", data)
    data_dict = data.dict(by_alias=True)
    print("data_dict : ", data_dict)
    TP2 = float(data_dict[ "TP2" ])
    TP3 = float(data_dict[ "TP3" ])
    H1 = data_dict[ "H1" ]
    DV_pressure = data_dict[ "DV_pressure" ]
    Reservoirs = data_dict[ "Reservoirs" ]
    Oil_temperature = data_dict[ "Oil_temperature" ]
    Motor_current = data_dict[ "Motor_current" ]
    COMP = data_dict[ "COMP" ]
    DV_eletric = data_dict[ "DV_eletric" ]
    Towers = data_dict[ "Towers" ]
    MPG = data_dict[ "MPG" ]
    LPS = data_dict[ "LPS" ]
    Pressure_switch = data_dict[ "Pressure_switch" ]
    Oil_level = data_dict[ "Oil_level" ]
    Caudal_impulses = data_dict[ "Caudal_impulses" ]
    # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction = model.predict([[TP2, TP3, H1, DV_pressure, Reservoirs, Oil_temperature, Motor_current, COMP,DV_eletric, Towers, MPG, LPS, Pressure_switch, Oil_level, Caudal_impulses]])
    if(prediction[0] == 1):
        prediction="Situation under Control"
    else:
        prediction="Maintainance Required"
    return {
        'prediction': prediction
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
#uvicorn app:app --reload
