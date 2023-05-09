import logging
import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.helper import get_constants

logger = logging.getLogger(__name__)


class NASDAQModel:
    def __init__(self, version: str) -> None:
        constants = get_constants()
        if "model-dir" not in constants.keys():
            logger.error("model-dir key needed in constants yaml")
            raise AttributeError("No model-dir in constants yaml")

        self.model_dir = constants["model-dir"]
        os.makedirs(self.model_dir, exist_ok=True)
        self.processed_dir = constants["processed-dir"]
        self.version = version
        self.model = None
        self.standard_scaler = None

    def pre_process(self):
        """
        Preprocess the dataset for fitting

        Returns:
            x_train: pd.DataFrame, training features
            y_train, pd.DataFrame, training labels
            x_test: pd.DataFrame, test features NOTE: needs(scaling)
            y_test: pd.DataFrame, test labels
        """
        if not os.path.exists(os.path.join(self.processed_dir, f"nasdaq_v{self.version}.parquet")):
            logger.error(f"Needs the processed file version {self.version} to train model on !!")
            raise FileNotFoundError(f"No processed file version {self.version} to train model on.")

        df = pd.read_parquet(os.path.join(self.processed_dir, f"nasdaq_v{self.version}.parquet"))
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # drop NaNs created after rolling average for the first few entries where,
        # we do not have a possible MVA or rolling average
        df.dropna(inplace=True)

        # Select features and target
        features = ['vol_moving_avg', 'adj_close_rolling_med']
        target = 'Volume'

        x = df[features]
        y = df[target]

        # split the dataset
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2023)

        return x_train, y_train, x_test, y_test  # x_test needs scaling

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame):
        """
        Train the model and saves it into a pkl file

        Args:
            x_train: pd.DataFrame, training features
            y_train: pd.DataFrame, training labels

        Returns:
            None
        """
        # normalize the data
        self.standard_scaler = StandardScaler()
        logger.info("Fitting the scalar on the training data !!")
        x_train = self.standard_scaler.fit_transform(x_train, y_train)

        # saving the scalar
        pickle.dump(self.standard_scaler, open(os.path.join(self.model_dir, f"SScaler_dv{self.version}.pkl"), 'wb'))

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Train the model
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_train)

        logger.info("-" * 10)
        logger.info(f"Training MAE is : {round(mean_absolute_error(y_train, y_pred), 4)}")
        logger.info(f"Training MSE is : {round(mean_squared_error(y_train, y_pred), 4)}")
        logger.info(f"Training R2 is : {round(r2_score(y_train, y_pred), 4)}")
        logger.info("-" * 10)

        # save the model
        pickle.dump(self.model, open(os.path.join(self.model_dir, f"RF_dv{self.version}.pkl"), 'wb'))

    def load(self):
        """
        Loads the model and scalar from the pkl files

        Returns:
            None
        """
        if os.path.exists(os.path.join(self.model_dir, f"RF_dv{self.version}.pkl")) and \
                os.path.exists(os.path.join(self.model_dir, f"SScaler_dv{self.version}.pkl")):
            self.model = pickle.load(open(os.path.join(self.model_dir, f"RF_dv{self.version}.pkl"), 'rb'))
            self.standard_scaler = pickle.load(
                open(os.path.join(self.model_dir, f"SScaler_dv{self.version}.pkl"), 'rb'))
        else:
            FileNotFoundError(f"Either ml model or scalar is not present")

    def test(self, x_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
        """
        Test the model fitted on the test set

        Args:
            x_test: pd.DataFrame, with features
            y_test: pd.DataFrame, with labels

        Returns:
            None
        """
        logger.info("Scaling the test data using the scalar !!")
        x_test = self.standard_scaler.transform(x_test)
        y_pred = self.model.predict(x_test)

        logger.info("-" * 10)
        logger.info(f"Test MAE is : {round(mean_absolute_error(y_test, y_pred), 4)}")
        logger.info(f"Test MSE is : {round(mean_squared_error(y_test, y_pred), 4)}")
        logger.info(f"Test R2 is : {round(r2_score(y_test, y_pred), 4)}")
        logger.info("-" * 10)

    def predict(self, vol_moving_avg: float, adj_close_rolling_med: float) -> float:
        df = pd.DataFrame()
        df["vol_moving_avg"] = [vol_moving_avg]
        df["adj_close_rolling_med"] = [adj_close_rolling_med]
        return self.model.predict(self.standard_scaler.transform(df))


if __name__ == "__main__":
    m = NASDAQModel(version="0")
    x_train, y_train, x_test, y_test = m.pre_process()
    m.fit(x_train, y_train)

    # load the saved model
    m.load()

    # test the trained model
    m.test(x_test, y_test)
