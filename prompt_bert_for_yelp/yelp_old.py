import json
import pandas as pd

# data_file = open("yelp_academic_dataset_checkin.json")

def create_path(name):
    return '/usr01/taku/Downloads/' + name

def read_data(path):
    data_file = open(path)
    data = []
    for line in data_file:
        data.append(json.loads(line))
    checkin_df = pd.DataFrame(data)
    data_file.close()
    return checkin_df


