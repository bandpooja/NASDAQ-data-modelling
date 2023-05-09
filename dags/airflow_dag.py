from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import gdown
import os
from pathlib import Path
import sys
import uuid
import zipfile

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loader.feature_engineering import pre_processing_parquet
from loader.standardizing import Standardizer
from model.dask_rf import NASDAQModel
from utils.helper import get_constants


version = str(uuid.uuid1())
constants = get_constants()
constants = {
    "root-dir": "pooja/data/nasdaq",
    "standardized-dir": "pooja/data/standardized",
    "processed-dir": "pooja/data/processed",
    "model-dir": "pooja/nasdaq_model"
}
# create the directories
for k in constants.keys():
    Path(constants[k]).mkdir(parents=True, exist_ok=True)


def download_data():
    # Download the kaggle data
    url = "https://drive.google.com/uc?id=16Ox02PqO3JibvB1cMkdH1mqfc5Sp8KGm"
    output = os.path.join(constants["root-dir"], "archive.zip")
    gdown.download(url, output, quiet=False)

    # unzip it in the root-dir
    with zipfile.ZipFile(os.path.join(constants["root-dir"], "archive.zip"), 'r') as zip_ref:
        zip_ref.extractall(constants["root-dir"])

    # remove the original zip
    os.remove(os.path.join(constants["root-dir"], "archive.zip"))


def check_raw_data():
    assert os.path.exists(constants["root-dir"]), f'Root directory {constants["root-dir"]}, in the constants.yaml ' \
                                                  f'should exist !!'
    assert os.path.exists(os.path.join(constants["root-dir"], "symbols_valid_meta.csv")), "Master csv doesnt exist"


def standardize():
    """
    Standardize the raw data to parquets for quick queries and modelling
    """
    s = Standardizer()
    s.standardize_via_multiprocessing(version=version, n_workers=4)


def check_standardized():
    assert os.path.exists(os.path.join(constants["standardized-dir"], f"nasdaq_v{version}.parquet")), "Standardized " \
                                                                                                      "file does not " \
                                                                                                      "exist"


def pre_process():
    """
    Perform feature engineering on the standardized data to have features used for modelling
    """
    assert "standardized-dir" in constants.keys(), f"A directory to save the processed data is needed in " \
                                                   f"constants.yaml !!"
    pre_processing_parquet(standardized_dir=constants["standardized-dir"], version="1",
                           processed_dir=constants["processed-dir"])


def check_preprocess():
    assert os.path.exists(os.path.join(constants["processed-dir"], f"nasdaq_v{version}.parquet")), "processed file " \
                                                                                                   "does not exist"


def train():
    """
    Trains the model on the processed data
    """
    m = NASDAQModel(version="1")
    x_train, y_train, x_test, y_test = m.pre_process()
    m.fit(x_train, y_train)

    # load the saved model
    m.load()

    # test the trained model
    m.test(x_test, y_test)


def check_model():
    assert os.path.exists(os.path.join(constants["model-dir"], f"SScaler_dv{version}.pkl")), "Scalar doesn't exist"
    assert os.path.exists(os.path.join(constants["model-dir"], f"RF_dv{version}.pkl")), "Model doesn't exist"


default_args = {
    "owner": "bandpooja",
    "depends_on_past": False,
    "start_date": datetime(2023, 5, 8),
    "email": ["airflow@airflow.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id='NASDAQ-ml',
    default_args=default_args,
    description='stock data ml DAG',
    schedule_interval="@monthly",
    start_date=datetime(2023, 5, 8)
) as dag:
    get_raw = PythonOperator(
        task_id="get-raw-data",
        python_callable=download_data
    )

    check_raw = PythonOperator(
        task_id="check-raw-data",
        python_callable=check_raw_data
    )

    standardize_data = PythonOperator(
        task_id='standardize-data',
        python_callable=standardize
    )

    check_data = PythonOperator(
        task_id="check-data",
        python_callable=check_standardized
    )

    pre_process_data = PythonOperator(
        task_id='feature-engineering',
        python_callable=pre_process
    )

    check_fe = PythonOperator(
        task_id="check-pprocess",
        python_callable=check_preprocess
    )

    train_model = PythonOperator(
        task_id='train-model',
        python_callable=train
    )

    check_model_pkls = PythonOperator(
        task_id="check-model",
        python_callable=check_model
    )

    host_model = BashOperator(
        task_id="host-model",
        bash_command="uvicorn plugins.app.predictor_app:app --reload --port 8080 --host 0.0.0.0"
    )

    get_raw >> check_raw >> standardize_data >> check_data >> pre_process_data >> check_fe >> train_model >> check_model_pkls >> host_model
