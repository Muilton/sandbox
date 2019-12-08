import json

def to_json(func):
    def wrapped():
        result = json.dumps(func())
        return result
    return wrapped

@to_json
def get_data():
    return {'data': 42}
  
get_data()  # вернёт '{"data": 42}'