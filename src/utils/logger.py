# module_logger.py
import logging

logger = logging.getLogger("my_agent")
logger.setLevel(logging.INFO)  # Or DEBUG for verbose
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)
