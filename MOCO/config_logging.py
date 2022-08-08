import logging
import os

def setup_logs(save_dir, run_name):
    """
    save_dir: saving directory.
    run_name: file name of log file.
    """
    # initialize logger
    logger = logging.getLogger("Moco")
    logger.setLevel(logging.INFO)

    # Determine the existence of save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # create the logging file handler
    log_file = os.path.join(save_dir, run_name + ".log")
    fh = logging.FileHandler(log_file) # control the log file output

    # create the logging console handler
    ch = logging.StreamHandler() # control the console sys

    # format
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    fh.setFormatter(formatter)

    # add handlers to logger object
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger