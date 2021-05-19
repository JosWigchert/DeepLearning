import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

import pickle
print('Starting')
Y_test = []
Y_pred = []
time = []

f = open('E:/log.txt', encoding="utf8")
lines = f.readlines()
f.close()
# splitting data and adding to arrays
print('Crunching data ...')
for i,line in enumerate(lines):
    try:
        line = line.replace('\x00', '')
        split = line.split(';')
        if (len(split) == 2):
            i += 1
            time.append(i)
            Y_pred.append(int(split[0]))
            Y_test.append(int(split[1]))
    except:
        print('error at line number: ',i )

names = ['Walking', 'Running', 'Cycling', 'Climbing Stairs']
uniqueTest = np.unique(Y_test)
uniquePred = np.unique(Y_pred)
maxTypes = max(len(uniqueTest), len(uniquePred))
mat = confusion_matrix(Y_test, Y_pred)
plot_confusion_matrix(conf_mat=mat, class_names=names[0:maxTypes], show_normed=True, figsize=(maxTypes,maxTypes))
plt.show()

plt.plot(time, Y_pred, label = "Type")
plt.xlim(0, 20000)
plt.show()
print('Done')