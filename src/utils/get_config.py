# import yaml
# import os

# def get_config(config_path=None):
#     # Step 1: Default config path relative to this file (utils or common)
#     if config_path is None:
#         script_dir = os.path.dirname(os.path.abspath(__file__))  # utils/ or wherever this file is
#         config_path = os.path.join(script_dir, "../common/config.yaml")

#     # Step 2: Absolute path to config
#     config_path = os.path.abspath(config_path)

#     # Step 3: Load YAML
#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)

#     # Step 4: Project root is two levels up from common/
#     project_root = os.path.abspath(os.path.join(config_path, "../../"))  # project_root/

#     # Step 5: Resolve all paths relative to project root
#     config["results"]["OUTPUT_PATH"] = os.path.join(project_root, "src", config["results"]["OUTPUT_PATH"])
#     os.makedirs(config["results"]["OUTPUT_PATH"], exist_ok=True)

#     config["data"]["DATASET_PATH"] = os.path.join(project_root, "src", config["data"]["DATASET_PATH"])
#     config["llm"]["api_key_path"] = os.path.join(project_root, "src", config["llm"]["api_key_path"])

#     return config

# # Usage
# config = get_config()

# DATASET_PATH = config["data"]["DATASET_PATH"]
# DATASET_NAME = config["data"]["DATASET_NAME"]

# OUTPUT_PATH = config["results"]["OUTPUT_PATH"]

# openai_api_key_path = config["llm"]["api_key_path"]
# with open(openai_api_key_path, "r") as f:
#     OPENAI_API_KEY = f.read().strip()  # remove any trailing newlines/spaces

# LLM_MODEL = config["llm"]["model"]
# llm_temperature = config["llm"]["temperature"]
# llm_max_tokens = config["llm"]["max_tokens"]

import yaml
import os
import getpass
from dotenv import load_dotenv
load_dotenv()  # loads .env automatically

def get_config(config_path=None):
    """
    Load config.yaml and resolve all paths relative to project_root.
    Creates results and logs folders automatically.
    """
    # Default config path (under src/common/)
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))  # src/utils/ or wherever this file is
        config_path = os.path.join(script_dir, "../common/config.yaml")

    config_path = os.path.abspath(config_path)

    # Load YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Compute project_root (parent of src/)
    project_root = os.path.abspath(os.path.join(os.path.dirname(config_path), "../.."))

    # Resolve paths relative to project_root
    # Dataset (under src/)
    config["data"]["DATASET_PATH"] = os.path.join(project_root, config["data"]["DATASET_PATH"])

    # Results and logs (under src/)
    config["results"]["OUTPUT_PATH"] = os.path.join(project_root, config["results"]["OUTPUT_PATH"])
    config["results"]["log_path"] = os.path.join(project_root, config["results"]["log_path"])
    os.makedirs(config["results"]["OUTPUT_PATH"], exist_ok=True)
    os.makedirs(config["results"]["log_path"], exist_ok=True)

    # API key (under CodeBase/)
    # config["llm"]["api_key_path"] = os.path.join(project_root, config["llm"]["api_key_path"])

    return config

# Usage
config = get_config()

DATASET_PATH = config["data"]["DATASET_PATH"]
DATASET_NAME = config["data"]["DATASET_NAME"]

OUTPUT_PATH = config["results"]["OUTPUT_PATH"]
LOG_PATH = config["results"]["log_path"]

# with open(config["llm"]["api_key_path"], "r") as f:
#     OPENAI_API_KEY = f.read().strip()

LLM_MODEL = config["llm"]["model"]
LLM_TEMPERATURE = config["llm"]["temperature"]
LLM_MAX_TOKENS = config["llm"]["max_tokens"]

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    OPENAI_API_KEY = getpass.getpass("Enter your OpenAI API key: ").strip()

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key is required. Set it via the OPENAI_API_KEY environment variable or input prompt.")