with open("config_example.toml", "rb") as f:
    content = f.read().decode("utf-8", errors="ignore")
with open("config_example_clean.toml", "w", encoding="utf-8") as f:
    f.write(content)



import toml
import logging

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger()

try:
    with open("config_example.toml", "r", encoding="utf-8") as f:
        config = toml.load(f)
except Exception as e:
    logger.error(f"Error loading configuration: {e}")