import sys

from utils.cfg import cfg

# 日志部分
from loguru import logger
logger.remove()
logger.add(sys.stderr, level=cfg.get("General", "LogLevel"))
logger.add("logs/app.log", level=cfg.get("General", "LogLevel"))

if __name__ == "__main__":
    logger.success(f"Successfully loaded log module. Current LogLevel: {cfg.get('General', 'LogLevel')}")