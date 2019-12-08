import json
import os
import tempfile
import argparse

def add(key, value):
    path = 'dictionary.json'
    _dict = {key: value}  
    
    data = []
    if os.path.exists(path):
        with open('dictionary.json', 'r') as f:
            data = json.loads(f.read()) 
            
    data.append(_dict)
    with open('dictionary.json', 'w') as f:
        json.dump(data, f)
        
def find(key):
    with open('dictionary.json', 'r') as f:
        data = json.loads(f.read())
        
    data = list(filter(lambda x: list(x.keys())[0] == key, data))
    for i, d in enumerate(data):
        if i == len(data)-1:
            print(f'{d[key]}')
        else:
            print(f'{d[key]}, ', end='')

parser = argparse.ArgumentParser() 
parser.add_argument("--key") 
parser.add_argument("--value") 
args = parser.parse_args() 
        
if args.key and args.value:
    add(args.key, args.value)
elif args.key:
	find(args.key)
	
	
	
	