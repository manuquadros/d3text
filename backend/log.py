import logging

logger = logging.getLogger("backend")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(module)s:%(funcName)s - %(levelname)s - %(message)s"
)

ch.setFormatter(formatter)

logger.addHandler(ch)
