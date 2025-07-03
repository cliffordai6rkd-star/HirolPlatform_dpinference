import sys

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="DEBUG",
           format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level} | {name}: {message}</level>")
