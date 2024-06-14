from helper.py import mismatch
from helper.py import down_mapping
from helper.py import getembeddings
import os
import numpy as np

dir = '/content/drive/MyDrive/frontalface'   #replace with your image folder path here
#2 5 3 6 1 4
imgpaths = []
for path in os.listdir(dir):
  imgpaths.append(path)


embmatrix = getembeddings(imgpaths)

percentage, vector_length, anchor = mismatch(embmatrix)
#print("Accuracy = {}%".format(percentage))
print(anchor)



