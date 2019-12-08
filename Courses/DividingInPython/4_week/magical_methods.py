import os
import tempfile


class File:
    def __init__(self, path):
        self.path = path
        self.current_position = 0

    def read(self):
        with open(self.path, 'r') as f:
            return f.read()

    def write(self, content):
        with open(self.path, 'w') as f:
            f.write(content)

    def __add__(self, obj):
        new_file = type(self)(os.path.join(tempfile.gettempdir(), 'temp.txt'))
        new_file.write(self.read() + obj.read())
        return new_file

    def __str__(self):
        return self.path

    def __iter__(self):
        return self

    def __next__(self):
        with open(self.path, 'r') as f:
            f.seek(self.current_position)

            line = f.readline()

            if not line:
                self.current_position = 0
                raise StopIteration('EOF')

            self.current_position = f.tell()

            return line