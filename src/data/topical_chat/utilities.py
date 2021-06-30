import json


def load_json_from_path(path_to_json):
    """
    Simple utility to read a JSON file from a path
    :param path_to_json: Path to the json file to read
    :return: A Python dict object representing the deserialized JSON
    """
    with open(path_to_json, "r") as json_file:
        return json.load(json_file)
