# External modules
import os
import time
import logging
from pathlib import Path

# Custom modules
import config as cfg

# Setup logging and start a logging instance
cfg.setup_logging()
logger = logging.getLogger(__name__)


# Check if a file exists, return true or false
def fileExists(file):
    if Path(file).exists():
        return True
    else:
        return False


# Check if folder exist, if not create folder.
def checkCreateDir(dir):
    if not fileExists(dir):
        try:
            os.makedirs(dir)
            logger.info(f"Created folder. Path: {dir}")
        except Exception as e:
            logger.info(f"Error creating folder. Path: {dir}")


# Delete files in the output folder
def cleanupOutputFolder():
    dir_output = cfg.pathOutput
    counter = 0

    logger.info(f"Deleting files from output folders {dir_output} in...")

    countdown(5)

    for item in os.listdir(dir_output):
        if item.endswith(".xlsm"):
            file_path_output = os.path.join(dir_output, item)
            logger.info(f"Deleting {file_path_output}")
            os.remove(file_path_output)
            counter = counter + 1

    logger.info(f"Deleted {counter} files.")


# Countdown function
def countdown(i):
    for x in range(i):
        print(i)
        time.sleep(1)
        i = i - 1
