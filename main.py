from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib 

import joblib
model = joblib.load('Liner_Regression_model.joblib')
scaler = joblib.load('scaler.joblib') 

app = FastAPI()
# GET request
@app.get("/")
def read_root():
 return {"message": "Welcome to Tuwaiq Academy"}
# post request
class InputFeatures(BaseModel):
    appearance:int
    minutes_played:int
    award:int 
    days_injured:int
    games_injured:int
    highest_value:int



def preprocessing(input_features: InputFeatures):
    dict_f = {
        'appearance': input_features.appearance,
        'minutes_played': input_features.appearance,
        'award': input_features.award,
        'highest_value': input_features.highest_value,
        'days_injured' : input_features.days_injured,
        'games_injured' : input_features.games_injured

    }
# Convert dictionary values to a list in the correct order
    features_list = [dict_f[key] for key in sorted(dict_f)]
# Scale the input features
    scaled_features = scaler.transform([list(dict_f.values())])
    return scaled_features

@app.post("/predict")
async def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    y_pred = model.predict(data)
    return {"pred": y_pred.tolist()[0]}


@app.get("/items/{item_id}")

@app.post("/items/{item_id}")

def create_item(item_id: int):
 return {"item": item_id}
