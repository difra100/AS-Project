import json


def load_data(name):
    jfile = open(name, "r")
    dicti = json.load(jfile)
    return dicti