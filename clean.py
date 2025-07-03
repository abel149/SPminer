import pickle
from pprint import pprint

with open("metta.pkl", "rb") as f:
    data = pickle.load(f)

pprint(data)
