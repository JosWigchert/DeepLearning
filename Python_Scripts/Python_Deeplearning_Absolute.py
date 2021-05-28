
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
import math


f = open('SmallDataset.TXT', encoding="utf8")
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
lowestSampleCount = 13197

Walking = df[df['Activity'] == 0].head(lowestSampleCount).copy()
Running = df[df['Activity'] == 1].head(lowestSampleCount).copy()
Cycling = df[df['Activity'] == 2].head(lowestSampleCount).copy()
Stairs = df[df['Activity'] == 3].head(lowestSampleCount).copy()

balanced_data = pd.DataFrame()
balanced_data = balanced_data.append([Walking, Running, Cycling, Stairs])
print(balanced_data['Activity'].value_counts())
#print(balanced_data.head())

#label = LabelEncoder()
#balanced_data['label'] = label.fit_transform(balanced_data['Activity'])

def getAbsoluteData(data):
    squared = (data['X data']*data['X data'])  + (data['Y data']*data['Y data'])  +  (data['Z data']*data['Z data'])
    sqrt = squared.apply(np.sqrt)

    return sqrt

### Standardize data
balanced_data['X data'] = balanced_data['X data'].astype('float')#Convert X data to float
balanced_data['Y data'] = balanced_data['Y data'].astype('float')#Convert Y data to float
balanced_data['Z data'] = balanced_data['Z data'].astype('float')#Convert Z data to float

absol = getAbsoluteData(balanced_data[['X data','Y data','Z data']]).to_numpy()

X = pd.DataFrame({'Abs': absol}) 
print(X)

Y = balanced_data['Activity']

# Scale the input data
scaler = StandardScaler()
# X = scaler.fit_transform(X)

scaled_X = pd.DataFrame(data = X, columns = ['Abs'])
scaled_X['Actv_label']= Y.values
print(scaled_X)

### Make a frame 
Fs = 200 #Samples per second
Seconds = 2 # Seconds the frame covers
frame_size = Fs*Seconds
hop_size = Fs*1 # The movement of the frame this results in overlapping data
#print(hop_size)

def get_frames(df, frame_size, hop_size):

    N_FEATURES = 1 # Amount of inputs 3 because x, y, z,

    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        x = df['Abs'].values[i: i + frame_size]
        
        # Retrieve the most often used label in this segment
        label = stats.mode(df['Actv_label'][i: i + frame_size])[0][0]
        frames.append([x])
        labels.append(label)

    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)

    return frames, labels

X, y = get_frames(scaled_X, frame_size, hop_size)


# Here we are dividing the data into training data and test data using train_test_split() from sklearn 
# which we have already imported. We are going to use 80% of the data for training the model and 20% of the data for testing. 
# random_state controls the shuffling applied to the data before applying the split. stratify = y splits the data in a stratified fashion, using y as the class labels.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)

print(X_train.shape)#Prints train dataset size
print(X_test.shape)#Prints test dataset size

# A CNN only accepts 3 dimentional data so we are going to reshape() our data and just a dimension with a constant value.
X_train = X_train.reshape(209, 400)
X_test = X_test.reshape(53, 400)

print(y_train.shape)
## Create the model
model = Sequential()

model.add(Dense(64, activation = 'relu', input_dim=400))

model.add(Dense(64, activation = 'relu'))

model.add(Dense(4, activation='softmax'))

iterations = 250

#Here we are compiling the model and fitting it to the training data. We will use 200 epochs to train the model.
#An epoch is an iteration over the entire data provided. validation_data is the data on which to evaluate the loss and any model metrics at the end of each epoch.
#The model will not be trained on this data. As metrics = ['accuracy'] the model will be evaluated based on the accuracy.
model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(X_train, y_train, epochs = iterations, validation_data= (X_test, y_test), verbose=1)

def plot_learningCurve(history, epochs):
    # Plot training & validation accuracy values
    epoch_range = range(1, epochs+1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

plot_learningCurve(history, iterations)

from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

y_pred = model.predict_classes(X_test)
array = ['Walking', 'Running', 'cycling', 'stairs']
mat = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_mat=mat, class_names= array, show_normed=True, figsize=(3,3))
plt.show()