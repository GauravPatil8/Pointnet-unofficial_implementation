import yaml

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
