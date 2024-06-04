import sys

from loguru import logger

FORMAT = (
    "<green>{time:MM-DD HH:mm:ss.S}</green> | "
    "<level>{level.icon}</level> | "
    "<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
COMPILE_FORMAT = "<green>{time:MM-DD HH:mm:ss.S}</green> | <level>{message}</level>"

log = logger


def set_logger_format():
    log.remove()
    log.add(sys.stdout, format=FORMAT, level="TRACE", filter=lambda record: "compile_log" not in record["extra"])
    log.add(sys.stdout, format=COMPILE_FORMAT, level="TRACE", filter=lambda record: "compile_log" in record["extra"])

    levels = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
    for level_name in levels:
        log.level(level_name, icon=level_name[0])
