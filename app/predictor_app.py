from fastapi import FastAPI
from fastapi.exceptions import HTTPException
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from model.dask_rf import NASDAQModel


app = FastAPI(
    version="1.0.0",
    description="An endpoint to make volume prediction using the model trained on NASDAQ data, the prediction is made "
                "using the volume moving average and adjacent close rolling median. \n"
                "The model fitted on the dataset is a Random Forest Regressor with parallelized training using dask."
)
model_dir = "C:\\Users\\bandp\\Documents\\Data_Engineering\\model_dir"
cwd = os.getcwd()

model = NASDAQModel(version="1")
# loading the model only once
model.load()


def model_prediction(moving_avg: float, rolling_med: float):
    prediction = model.predict(vol_moving_avg=moving_avg, adj_close_rolling_med=rolling_med)
    return prediction


@app.get("/")
async def home():
    return {
        "status": "SUCCESSFUL",
        "information": "Running the volume prediction application"
    }


@app.get("/predict")
async def predict_volume(vol_moving_avg: float = None, adj_close_rolling_med: float = None):
    try:
        prediction = model_prediction(vol_moving_avg, adj_close_rolling_med)
        return {
            "status": "SUCCESSFUL",
            "prediction": {
                "volume": prediction
            }
        }
    except Exception as e:
        return HTTPException(status_code=411, detail=f"Error making prediction, error {e}")
