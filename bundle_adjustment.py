from __future__ import print_function
import urllib.request
import bz2
import os
import numpy as np


if __name__ == "__main__":
    dataset_url = "http://grail.cs.washington.edu/projects/bal/data/ladybug/problem-49-7776-pre.txt.bz2"
    filename = "dataset.txt.bz2"
    urllib.request.urlretrieve(dataset_url, filename)
