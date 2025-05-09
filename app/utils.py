import yaml

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return None


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def dict_to_object(data):
    if isinstance(data, dict):
        return Config(**data)
    return data
