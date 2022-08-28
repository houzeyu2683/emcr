
import pandas
import os

class table:

    def __init__(self, path=None, name=None):

        self.path = path
        self.name = name if(name) else [os.path.basename(p) for p in self.path]
        loop = zip(self.path, self.name)
        self.sheet = [(n, pandas.read_csv(p, dtype=str)) for p, n in loop]
        return

    pass
