import logging
from logging.handlers import RotatingFileHandler
import os

logger = logging.getLogger(__name__)


def crash_report(description):
    logger.info("crash: {}".format(description))
    
    # https://stackoverflow.com/a/10645855
    logging.error("exception ", stack_info=True, exc_info=True)


# directory to write the file in
project_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(project_dir, '../pb-nasdaq-task.log')

# Based on https://stackoverflow.com/a/22313803
logging.addLevelName(logging.INFO, "")
logging.addLevelName(logging.DEBUG, "")

logging_level = logging.INFO

logging.getLogger().setLevel(logging_level)

log_formatter = logging.Formatter("%(asctime)s, %(levelname)s: - file: %(filename)s, %(lineno)d - %(message)s")

try:
    logger_file_handler = RotatingFileHandler(
        log_file, maxBytes=10485760, backupCount=4
    )
    logger_file_handler.setLevel(logging_level)
    logger_file_handler.setFormatter(log_formatter)
    logging.getLogger().addHandler(logger_file_handler)
except:
    pass

logger_console_handler = logging.StreamHandler()
logger_console_handler.setLevel(logging_level)
logger_console_handler.setFormatter(log_formatter)
logging.getLogger().addHandler(logger_console_handler)
