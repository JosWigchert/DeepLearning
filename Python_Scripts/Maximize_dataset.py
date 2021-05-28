import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
print(tf.version)

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import scipy.stats as stats


f = open('SmallDataset.txt', encoding="utf8")
lines = f.readlines()
f.close()


# initializing arrays
x=[]
y=[]
z=[]
time=[]
type=[]
i=0


# splitting data and adding to arrays
print('Crunching data ...')
for i,line in enumerate(lines):
    try:
        line = line.replace('\x00', '')
        split = line.split(',')
        if (len(split) >= 4):
            i += 1
            x.append(int(split[0]))
            y.append(int(split[1]))
            z.append(int(split[2]))
            time.append(i)
            type.append(int(split[3])) #* 500)
    except:
        print('error at line number: ',i )

df = pd.DataFrame() 
df['Activity'] = type
#df['time'] = np.arange(0,len(x))
df['X data'] = x
df['Y data'] = y
df['Z data'] = z

#check for null values in dataset
#print(df.isnull().sum())

#Check activity distribution
print(df['Activity'].value_counts())

#Balance the dataset
df['X data'] = df['X data'].astype('float')#Convert X data to float
df['Y data'] = df['Y data'].astype('float')#Convert Y data to float
df['Z data'] = df['Z data'].astype('float')#Convert Z data to float
#df.info()

# time between measurements is 5 ms thus Fs = 200

activities = df['Activity'].value_counts().index

#Switch codes Walking = 00 Running = 01  Cycling = 10 Stairs= 11
#Walking = 0
#Running = 1
#Cycling = 2
#Stairs = 3

#Smallest dataset is 9540 samples(for walking) thus we will only use the same amount of samples for each activity 
lowestSampleCount = 30365

Walking = df[df['Activity'] == 0].head(lowestSampleCount).copy()
Running = df[df['Activity'] == 1].copy()


#print(Walking)


# Copy from this part to lower fs from 200 to 50
NewWalking = pd.DataFrame(columns=['Activity', 'X data', 'Y data', 'Z data'])
c = 0
for x in range(0,lowestSampleCount,4):
  NewWalking.loc[c] = Walking.iloc[x]
  c = c + 1  
for x in range(1,lowestSampleCount,4):
  NewWalking.loc[c] = Walking.iloc[x]
  c = c + 1  
for x in range(2,lowestSampleCount,4):
  NewWalking.loc[c] = Walking.iloc[x]
  c = c + 1  
for x in range(3,lowestSampleCount,4):
  NewWalking.loc[c] = Walking.iloc[x]
  c = c + 1  

c = 0
NewRunning = pd.DataFrame(columns=['Activity', 'X data', 'Y data', 'Z data'])
for x in range(0,lowestSampleCount,4):
  NewRunning.loc[c] = Running.iloc[x]
  c = c + 1  
for x in range(1,lowestSampleCount,4):
  NewRunning .loc[c] = Running.iloc[x]
  c = c + 1  
for x in range(2,lowestSampleCount,4):
  NewRunning.loc[c] = Running.iloc[x]
  c = c + 1  
for x in range(3,lowestSampleCount,4):
  NewRunning .loc[c] = Running.iloc[x]
  c = c + 1  

#print(NewWalking)


balanced_data = pd.DataFrame()
balanced_data = balanced_data.append([NewWalking, NewRunning])

balanced_data.to_csv(r'balanced_data_50hz.csv',index = False)

print(balanced_data['Activity'].value_counts())




