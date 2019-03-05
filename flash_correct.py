import os
import re

import numpy as np
import matplotlib.pyplot as plt
import cv2

def tryint(s):#This is for sorting the files into human numerical order
    try: return int(s)
    except: return s

def alphanum_key(s):#This is for sorting the files into human numerical order
    return [tryint(c) for c in re.split("([0-9]+)", s)]

plt.ion()

os.chdir("c:/Users/fc12293/Chrome Local Downloads/gary")

file_extension = ".tiff"

files = [i for i in os.listdir() if i[-len(file_extension):] == file_extension]
files.sort(key = alphanum_key)

img = cv2.imread(files[8])
flash = cv2.imread(files[9])

fig, (ax1,ax2) = plt.subplots(2)
ax1.imshow(img)
ax2.imshow(flash)
plt.show()

grey1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grey2 = cv2.cvtColor(flash, cv2.COLOR_BGR2GRAY)
