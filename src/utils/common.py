import os
import yaml
import logging

def read_yaml(file_path):
    """ Reads yaml file and returns entire contents """
    try:
        with open(file_path) as yaml_file:
            content = yaml.safe_load(yaml_file)    
        return content
    except yaml.error.YAMLError as e:
        print("yaml error occured: ",e)
    except Exception as e:
        print("unexpected error occured: ", e)

def logger_init():
    os.makedirs(R"src/logs/", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(R"src/logs/info.log"),
            logging.StreamHandler()
        ]
    )