from functools import partial
import logging
from multiprocessing import Pool
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from utils.helper import get_constants

logger = logging.getLogger(__name__)


class Standardizer:
    def __init__(self) -> None:
        self.constants = get_constants()

        assert os.path.exists(self.constants[
                                  "root-dir"]), f'Root directory {self.constants["root-dir"]}, in the constants.yaml ' \
                                                f'should exist !!'
        assert "standardized-dir" in self.constants.keys(), f"A directory to save the standardized data is needed in " \
                                                            f"constants.yaml !!"
        logger.info("Desired keys to standardize data exist in the constants.yaml")

        self.master_df = self.load_master()

    def load_master(self) -> pd.DataFrame:
        """
        Loads the master csv file

        Returns:
            pd.DataFrame
        """
        try:
            master_csv = os.path.join(self.constants["root-dir"], "symbols_valid_meta.csv")
            master_df = pd.read_csv(master_csv)
            return master_df
        except Exception as e:
            logger.error(f"Keys exist but error loading files from the root directory, {e}")

    def load(self, version: str = "0") -> pd.DataFrame:
        """
        Loads the dataframe from a parquet file

        Args:
            version: str, version of the parquet file to load

        Returns:
            pd.DataFrame
        """
        f_path = os.path.join(self.constants["standardized-dir"], f"nasdaq_v{version}.parquet")
        if not os.path.exists(f_path):
            logger.warning(
                f'Standardized parquet file with version {version} does not exists in the {self.constants["standardized-dir"]}, can not load.')
            raise FileNotFoundError(f"standardized version {version} does not exist !")

        return pd.read_parquet(f_path)

    def standardize(self, version: str = "0") -> None:
        """
        Performs standardization and saves the data to a parquet file

        Args:
            version: str, version of the parquet file to generate

        Returns:
            None
        """
        if os.path.exists(os.path.join(self.constants["standardized-dir"], f"nasdaq_v{version}.parquet")):
            logger.warning(
                f'Combined parquet file with version {version} already exists in the {self.constants["standardized-dir"]}, either delete it or create a new version.')
            return

        df_merged = None
        for dir_name, _, files in os.walk(self.constants["root-dir"]):
            for file_ in tqdm(files, desc=f"Standardizing csv files in {dir_name}", total=len(files)):
                f_path = os.path.join(dir_name, file_)

                if f_path == os.path.join(self.constants["root-dir"], "symbols_valid_meta.csv"):
                    # skip this iteration and process the next file
                    continue

                df_stock_etf_csv = pd.read_csv(f_path)
                # symbol for the file is embedded in its file name
                _, f_name = os.path.split(f_path)
                symbol, _ = os.path.splitext(f_name)

                master_for_symbol = self.master_df.loc[self.master_df["Symbol"] == symbol]

                if len(master_for_symbol) == 0:
                    logger.warning(f"Cant find symbol in master csv for {f_path}, {symbol}")
                    # ignore this case because we can not get the stock or etl name for this 
                    continue

                assert len(master_for_symbol) == 1, f"More than one entries with symbol {symbol} in master csv !!"

                df_stock_etf_csv["Symbol"] = symbol
                df_stock_etf_csv["Security Name"] = master_for_symbol["Security Name"].values.tolist()[
                    0]  # asserted to have length 1 above

                # if the df_merged is None
                if df_merged is None:
                    df_merged = df_stock_etf_csv.copy(deep=True)
                else:
                    df_merged = pd.concat([df_merged, df_stock_etf_csv], axis=0)

        df_merged.to_parquet(
            os.path.join(self.constants["standardized-dir"], f"nasdaq_v{version}.parquet")
        )

        logger.info(
            f"Completed standardizing and writing to parquet file version {version} in "
            f"the {self.constants['standardized-dir']}."
        )

    def merge_parquets(self, parquet_dir: str, version: str) -> None:
        """
        Merges the parquet files for individual symbol parquet files

        Args:
            parquet_dir: str, directory where the individual parquet files are saved
            version: str, version of the parquet file

        Returns:
            None
        """
        data_dir = Path(parquet_dir)
        full_df = pd.concat(pd.read_parquet(p_f) for p_f in tqdm(data_dir.glob('*.parquet'), desc="Merging the parquet files"))
        full_df.to_parquet(os.path.join(self.constants["standardized-dir"], f"nasdaq_v{version}.parquet"))

    @staticmethod
    def process_a_file(f_path: str, master_df: pd.DataFrame, save_dir: str) -> None:
        """
        Loads and process a csv for a symbol

        Args:
            f_path: str, path of the csv for a symbol
            master_df: pd.DataFrame, master dataframe read from the root directory
            save_dir: str, path where the individual parquet is going to be saved

        Returns:
            None
        """
        df_stock_etf_csv = pd.read_csv(f_path)
        # symbol for the file is embedded in its file name
        _, f_name = os.path.split(f_path)
        symbol, _ = os.path.splitext(f_name)

        master_for_symbol = master_df.loc[master_df["Symbol"] == symbol]

        if len(master_for_symbol) == 0:
            logger.warning(f"Cant find symbol in master csv for {f_path}, {symbol}")
            # ignore this case because we can not get the stock or etl name for this
            return

        assert len(master_for_symbol) == 1, f"More than one entries with symbol {symbol} in master csv !!"

        df_stock_etf_csv["Symbol"] = symbol
        df_stock_etf_csv["Security Name"] = master_for_symbol["Security Name"].values.tolist()[
            0]  # asserted to have length 1 above

        df_stock_etf_csv.to_parquet(os.path.join(save_dir, f"{symbol}.parquet"))

    @staticmethod
    def pprocess_each_symbol(files: list, save_dir: str, master_df: pd.DataFrame, n_workers: int) -> None:
        """
        Multiprocessing for all the symbols in the dataset

        Args:
            files: list, list of file paths to process
            save_dir: str, path to save the parquet files in
            master_df: pd.DataFrame, master datafra,e
            n_workers: int, number of workers to be used for multiprocessing

        Returns:
            None
        """
        with Pool(n_workers) as pool:
            pool.map(partial(Standardizer.process_a_file, master_df=master_df, save_dir=save_dir), files)

    def standardize_via_multiprocessing(self, version: str = "1", n_workers: int = 1) -> None:
        """
        Standardizes the symbols via multiprocessing

        Args:
            version: str, version of the final parquet to save
            n_workers: int, number of workers for multiprocessing

        Returns:
            None
        """
        if os.path.exists(os.path.join(self.constants["standardized-dir"], f"nasdaq_v{version}.parquet")):
            logger.warning(
                f'Combined parquet file with version {version} already exists in the'
                f' {self.constants["standardized-dir"]}, either delete it or create a new version.')
            return

        symbol_files = []
        save_dir = os.path.join(self.constants["standardized-dir"], "per_symbol")
        os.makedirs(save_dir, exist_ok=True)
        for dir_name, _, files in os.walk(self.constants["root-dir"]):
            for file_ in tqdm(files, desc=f"Standardizing csv files in {dir_name}", total=len(files)):
                f_path = os.path.join(dir_name, file_)

                if f_path == os.path.join(self.constants["root-dir"], "symbols_valid_meta.csv"):
                    # skip this iteration and process the next file
                    continue

                symbol_files.append(f_path)

        # parallely process each symbol to create parquet files
        self.pprocess_each_symbol(files=symbol_files, save_dir=save_dir, master_df=self.master_df, n_workers=n_workers)
        # merge parquet files
        self.merge_parquets(parquet_dir=save_dir, version=version)


if __name__ == "__main__":
    s = Standardizer()
    # s.standardize(version="0")
    # df = s.load(version="0")
    # print(df.head())

    s.standardize_via_multiprocessing(version="1", n_workers=4)
    df = s.load(version="1")
    print(df.head(), len(df))
