import argparse
import os
import tempfile
import json

parser = argparse.ArgumentParser()
parser.add_argument("--key")
parser.add_argument("--value")
args = parser.parse_args()


data_add = {args.key : args.value}

storage_path = os.path.join(tempfile.gettempdir(), 'storage.data')

if args.key and args.value:
    with open("C:/Workspace/coursera/diving_in_python/second_week/guests.json",'r') as f:
        data = json.loads(f.read())
    data.update(data_add)
    with open("1_guests.json",'w') as f:
        f.write(json.dumps(data))
        
elif args.key:
    with open("1_guests.json",'r') as f:
        data_find = json.loads(f.read())
        print(data_find[args.key])


