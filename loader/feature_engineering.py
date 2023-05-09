import logging
import os
import pandas as pd

from utils.helper import get_constants

logger = logging.getLogger(__name__)


def pre_processing_parquet(standardized_dir: str, version: str, processed_dir: str) -> pd.DataFrame:
    os.makedirs(processed_dir, exist_ok=True)
    if os.path.exists(os.path.join(processed_dir, f"nasdaq_v{version}.parquet")):
        df = pd.read_parquet(os.path.join(processed_dir, f"nasdaq_v{version}.parquet"))
        logger.info(f"Processed parquet version {version} already exists, loading it")
        return df

    if not os.path.exists(os.path.join(standardized_dir, f"nasdaq_v{version}.parquet")):
        logger.error(f"Version {version}, standardized parquet does not exists.")
        raise FileNotFoundError(f"Version {version}, standardized parquet does not exists.")

    df = pd.read_parquet(os.path.join(standardized_dir, f"nasdaq_v{version}.parquet"))
    df['vol_moving_avg'] = df['Volume'].rolling(30).mean()
    df['adj_close_rolling_med'] = df['Adj Close'].rolling(30).median()
    df.to_parquet(os.path.join(processed_dir, f"nasdaq_v{version}.parquet"))
    return df


if __name__ == "__main__":
    constants = get_constants()
    assert "standardized-dir" in constants.keys(), f"A directory to save the processed data is needed in " \
                                                   f"constants.yaml !!"
    pre_processing_parquet(standardized_dir=constants["standardized-dir"], version="1",
                           processed_dir=constants["processed-dir"])
