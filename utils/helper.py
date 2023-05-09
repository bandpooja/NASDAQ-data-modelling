import logging
import os
import yaml

logger = logging.getLogger(__name__)


def load_yaml(f_path: str) -> dict:
    """
        Opens and loads a yaml file to give a dictionary

        Parameters:
            - f_path: yaml file path as string

        Returns:
            - {} , yaml file loaded as dictionary
    """
    try:
        with open(f_path) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
        return data
    except Exception as e:
        logger.error(f"Error loading the yaml file {f_path}, Error: {e}")


def save_yaml(data: dict, f_path: str) -> None:
    """
        Saves the dictionary as a yaml file

        Parameters:
            - data: dictionary to save in yml
            - f_path: yaml file path as string

        Returns:
            - None
    """
    try:
        with open(f_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        return None
    except Exception as e:
        logger.error(f"Error writing the yaml file {f_path}, Error: {e}")


def get_constants() -> dict:
    """
        Loads the constants for the project

        Returns:
         - {}, dictionary containing constants
    """
    try:
        if os.path.exists("constants.yaml"):
            consts = load_yaml("constants.yaml")
        else:
            creds_f_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "constants.yaml")
            consts = load_yaml(creds_f_path)
        
        return consts
    except Exception as e:
        logger.error(f"Error loading constants for the project, {e}")
        raise IOError(f"Error, {e}")

