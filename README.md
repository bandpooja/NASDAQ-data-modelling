# StockMarketData
Data ref: [data](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset)

# Note
### Airflow DAG
Host the model trained via DAG is also proving to be different because the model size is `16 GB` and I will need to pay extra to host the model, therefore 
I have hosted the model locally and shown the result in the `result.ipynb`

### Docker Container APP
To complete the docker file I need to download the final model into a cloud service and downlod it in the docker image,
the model itself is `16 GB` in size and then the scaler is in KB, but if we can upload it then write a `.py` file to download the model and scalar in the Docker file
Then we can download the model and use the docker file to create a container app to host the model.

# Getting Started
1. Install the `requirements.txt`  (based on Python 3.10)
   ```bash
        pip install -r requirements.txt
   ```
2. Run the application in debug mode
   ```bash
      uvicorn app.predictor_app:app --reload --port 8081
   ```
3. Check the OpenAPI specifications after hosting in [http://0.0.0.0/8081/docs](http://0.0.0.0/8081/docs)

# Deploying it as a container app
1. Building the docker image
    ```bash 
        docker build -t nasdaq-volume-predictor --no-cache .
    ```
2. Run the docker image 
   ```bash
      docker run -dp 8081:8081 nasdaq-volume-predictor
   ```

# Run the dag 
1. build the docker using compose
   ```bash
      docker-compose up --build
   ``` 
2. Check the [localhost:8080](https://0.0.0.0:8080)


# Test Results
![test-plot](static/test.png)

# Sample Request and response
This process of how to request the API once hosted is shown in the notebook 
![demo-notebook](request_response_play.ipynb)

# Local logs during training

```text
2023-05-06 11:24:29,491, : - file: standardizing.py, 21 - Desired keys to standardize data exist in the constants.yaml
2023-05-06 11:29:29,963, : - file: standardizing.py, 21 - Desired keys to standardize data exist in the constants.yaml
2023-05-06 11:32:27,381, WARNING: - file: standardizing.py, 78 - Cant find symbol in master csv for C:\\Users\\transponster\\Documents\\pooja\\data\\nasdaq\stocks\AGM-A.csv, AGM-A
2023-05-06 11:35:07,065, WARNING: - file: standardizing.py, 78 - Cant find symbol in master csv for C:\\Users\\transponster\\Documents\\pooja\\data\\nasdaq\stocks\CARR#.csv, CARR#
2023-05-06 12:19:01,894, WARNING: - file: standardizing.py, 78 - Cant find symbol in master csv for C:\\Users\\transponster\\Documents\\pooja\\data\\nasdaq\stocks\UTX#.csv, UTX#
2023-05-06 12:25:16,495, : - file: standardizing.py, 99 - Completed standardizing and writing to parquet file version 0 in the C:\\Users\\transponster\\Documents\\pooja\\data\\standardized.
2023-05-06 14:43:21,513, : - file: standardizing.py, 24 - Desired keys to standardize data exist in the constants.yaml
2023-05-06 14:48:59,721, : - file: standardizing.py, 25 - Desired keys to standardize data exist in the constants.yaml
2023-05-06 14:49:08,188, WARNING: - file: standardizing.py, 154 - Cant find symbol in master csv for C:\\Users\\transponster\\Documents\\pooja\\data\\nasdaq\stocks\CARR#.csv, CARR#
2023-05-06 14:49:12,207, WARNING: - file: standardizing.py, 154 - Cant find symbol in master csv for C:\\Users\\transponster\\Documents\\pooja\\data\\nasdaq\stocks\AGM-A.csv, AGM-A
2023-05-07 11:00:00,127, : - file: dask_rf.py, 79 - Fitting the scalar on the training data !!
2023-05-07 11:50:22,484, : - file: dask_rf.py, 96 - ----------
2023-05-07 11:50:23,809, : - file: dask_rf.py, 97 - Training MAE is : 286389.2948
2023-05-07 11:50:24,024, : - file: dask_rf.py, 98 - Training MSE is : 10130539535362.332
2023-05-07 11:50:24,422, : - file: dask_rf.py, 99 - Training R2 is : 0.9373
2023-05-07 11:50:24,422, : - file: dask_rf.py, 100 - ----------
2023-05-07 12:17:22,690, : - file: dask_rf.py, 134 - Scaling the test data using the scalar !!
2023-05-07 12:23:39,755, : - file: dask_rf.py, 138 - ----------
2023-05-07 12:23:40,380, : - file: dask_rf.py, 139 - Test MAE is : 490354.5868
2023-05-07 12:23:40,430, : - file: dask_rf.py, 140 - Test MSE is : 40829722444872.24
2023-05-07 12:23:40,543, : - file: dask_rf.py, 141 - Test R2 is : 0.7082
2023-05-07 12:23:40,543, : - file: dask_rf.py, 142 - ----------
```