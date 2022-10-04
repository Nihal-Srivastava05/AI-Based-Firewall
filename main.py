import time
import numpy as np
import pickle

LAST_LINE = 0
while(True):
    time.sleep(1)
    f = open("./Datasets/log_file.txt", "r")
    data = f.readlines()[LAST_LINE:]
    for d in data:
        content = d.split(' ')

    LAST_LINE += len(data)