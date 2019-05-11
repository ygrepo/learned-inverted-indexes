import os
import numpy as np


class Collection:
    def __init__(self, collection_name):
        BASE_DIR = os.getcwd()
        collection_dir = os.path.join(BASE_DIR, collection_name)
        with open(collection_dir + ".docs", "rb") as docs_file:
            self.docs = np.fromfile(docs_file, dtype=np.uint32)
        with open(collection_dir + ".freqs", "rb") as freqs_file:
            self.freqs = np.fromfile(freqs_file, dtype=np.uint32)

    def __iter__(self):
        i = 2
        while i < len(self.docs):
            size = self.docs[i]
            yield (self.docs[i+1:size+i+1], self.freqs[i+1:size+i+1])
            i += size+1

    def __next__(self):
        return self


def main():
    test_collection = Collection("test_data/test_collection")
    for idx, a in enumerate(test_collection):
        print(idx, a)


if __name__ == "__main__":
    main()

