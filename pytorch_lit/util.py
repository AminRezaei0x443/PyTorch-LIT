import json


def read_json(file):
    f = open(file, mode="r")
    res = json.load(f)
    f.close()
    return res


def write_json(file, obj):
    f = open(file, mode="w")
    json.dump(obj, f)
    f.close()
