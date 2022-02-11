import pandas as pd
from pymongo import MongoClient

client = MongoClient()
database = client['BigData']
collection = database['predictData']
collection.drop()
collection = database['predictData']

def csv_to_json(filename):
    data = pd.read_csv(filename)
    data = data.loc[:, ['Latitude', 'Longitude', 'PrixNuitee']]
    return data.to_dict('records')

collection.insert_many(csv_to_json('./predict.csv'))
collection.update_many({},([ 
      {"$addFields": {
    "coordinate": [ "$Longitude", "$Latitude" ]
  }}
]))