class FileReader:
    def __init__(self, path):
        self.path = path
    
    def read(self):
        try:
            with open(f'{self.path}', 'r') as f:
                return f.read()
        except IOError:
            return ""
