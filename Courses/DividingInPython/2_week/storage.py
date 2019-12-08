import argparse
import os
import tempfile
import json

parser = argparse.ArgumentParser()
parser.add_argument("--key")
parser.add_argument("--val")
args = parser.parse_args()
data_args = {args.key : args.val}
data = []
storage_path = os.path.join(tempfile.gettempdir(), 'guests.json')


if args.key and args.val:
    if os.path.isfile(storage_path):
        with open(storage_path, 'r') as f:
            data = json.loads(f.read())
        data.append(data_args)
    else:
        data.append(data_args)

    with open(storage_path, 'w') as f:
        f.write(json.dumps(data))

elif args.key:
    if os.path.isfile(storage_path):
        with open(storage_path, 'r') as f:
            data_find = json.loads(f.read())
            for i in data_find:
                if args.key in list(i.keys()):
                    data.append(i[args.key])
        print(', '.join(str(x) for x in data))
    else:
        print(None)




