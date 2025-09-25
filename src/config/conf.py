import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_FILE = ROOT_DIR / "conf.yaml"

def load_config(section = "competencia01") -> dict:
    section_config = {}
    try:
        with open(CONFIG_FILE, 'r') as file:
            config = yaml.safe_load(file) or {}

        section_config = config.get(section, {})

        if not section_config:
            logger.warning(f"Section '{section}' not found in config. Returning empty dict.")

        logger.info(f"Config loaded successfully from {CONFIG_FILE}")

    except FileNotFoundError:
        logger.error(f"Config file not found: {CONFIG_FILE}")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config at {CONFIG_FILE}: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error loading config from {CONFIG_FILE}: {e}")
    
    return section_config
# import yaml
# import os
# import logging

# logger = logging.getLogger(__name__)

# PATH_CONFIG = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), "..", "..", "conf.yaml")
# )

# try:
#     with open(PATH_CONFIG, 'r') as file:
#         config = yaml.safe_load(file)
#         competencia01_config = config['competencia01']



# except Exception as e:
#     logger.error(f"Error al cargar el archivo de configuración: '{PATH_CONFIG}': {e}")


