# 配置文件
from configparser import ConfigParser
cfg = ConfigParser()

# 尝试读取配置文件
try:
    cfg.read(["./config.cfg", "../config.cfg"], encoding="utf-8")
except Exception as e:
    print(e)
    exit(1)