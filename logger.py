import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

FORMATTER = logging.Formatter(
    "%(levelname)s %(name)s %(asctime)s |" " %(filename)s:%(lineno)d | %(message)s"
)

# Configure console logger
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(FORMATTER)
# attach console handler to logger
logger.addHandler(console_handler)
