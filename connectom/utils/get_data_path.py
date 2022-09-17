import glob
from typing import List
import os


class LoadConnectomData:
    def __init__(self, path: str, type_data: str = "*.graphml") -> List[str]:
        try:
            os.path.isdir(path)
        except IOError:
            print("Directory not accessible")
        self.path = path
        self.type_data = type_data
        self.files = glob.glob(os.path.join(self.path, self.type_data))

    def __call__(self, idx=None):
        if idx is None:
            return self.files
        else:
            return self.files[idx]

    def __len__(self):
        return len(self.files)
