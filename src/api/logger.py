from __future__ import annotations

from copy import copy
import logging
import colorlog


class CustomFormatter(colorlog.ColoredFormatter):
    def formatMessage(self, record: logging.LogRecord) -> str:
        recordcopy = copy(record)
        separator = " " * (9 - len(recordcopy.levelname))
        recordcopy.__dict__["separator"] = separator
        return super().formatMessage(recordcopy)


handler = colorlog.StreamHandler()
formatter = CustomFormatter(
    (
        "%(log_color)s%(levelname)s%(white)s:%(separator)s%(message)s"
        " %(thin)s[%(asctime)s.%(msecs)03d]"
    ),
    log_colors={
        "DEBUG": "blue",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)
logger = colorlog.getLogger("speech_recognition_inference")
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
