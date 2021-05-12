import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt 
import glob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
print(tf.version)

FREQUENCY = 200
TARGET_FREQUENCY = 50
DOWNSAMPLE = int(FREQUENCY/TARGET_FREQUENCY)
print(DOWNSAMPLE)

#mean = [-48.89537296, -468.66104616, 141.17755091]
#scale = [283.37936145, -419.61237855, 254.63491622]

mean = [0, 0, 0]
scale = [1024, 1024, 1024]

def preprocess_files(filenames):
    datasets = []

    for f in filenames:
        datasets.extend(preprocess_file(f))

    return datasets


def preprocess_file(filename):
    x = []
    y = []
    z = []
    t = []

    datasets = []

    print(filename)

    f = open(filename, "r")
    for i,line in enumerate(f):
        try:
            split = line.split(',')
            if (len(split) >= 4) and i > FREQUENCY*10:
                x.append((int(split[0]) - mean[0])/scale[0])
                y.append((int(split[1]) - mean[1])/scale[1])
                z.append((int(split[2]) - mean[2])/scale[2])
                t.append(int(split[3]))
            else:
                # split data
                df = pd.DataFrame()
                df["type"] = t
                df["x"] = x
                df["y"] = y
                df["z"] = z
                x = []
                y = []
                z = []
                t = []

                # downsample the dataframe
                if len(df)/FREQUENCY > 5:
                    for j in range(DOWNSAMPLE):
                        downsampled = df.iloc[j::DOWNSAMPLE, :]
                        datasets.append(downsampled)
        except:
            print('error')
    f.close()

    return datasets

Fs = TARGET_FREQUENCY #Samples per second
Seconds = 2 # Seconds the frame covers
frame_size = Fs*Seconds
hop_size = 10 # The movement of the frame this results in overlapping data

def create_frames(df, frame_size, hop_size):
    N_FEATURES = 3 # Amount of inputs 3 because x, y, z,
    
    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        x = df['x'].values[i: i + frame_size]
        y = df['y'].values[i: i + frame_size]
        z = df['z'].values[i: i + frame_size]
        
        # Retrieve the most often used label in this segment
        label = stats.mode(df['type'][i: i + frame_size])[0][0]

        if len(x)==frame_size:
            frames.append([x, y, z])
            labels.append(label)

    # Bring the segments into a better shape
    frames = np.asarray(frames).transpose((0, 2, 1))
    labels = np.asarray(labels)

    return frames, labels

files = glob.glob("Datasets/walking/*.txt")
files.extend(glob.glob("Datasets/running/*.txt"))

datasets = preprocess_files(files)

X, y = create_frames(datasets[0], frame_size, hop_size)

for i in range(1,len(datasets)):
    temp_x, temp_y = create_frames(datasets[i], frame_size, hop_size)
    X = np.append(X, temp_x, axis=0)
    y = np.append(y, temp_y)

# Here we are dividing the data into training data and test data using train_test_split() from sklearn 
# which we have already imported. We are going to use 80% of the data for training the model and 20% of the data for testing. 
# random_state controls the shuffling applied to the data before applying the split. stratify = y splits the data in a stratified fashion, using y as the class labels.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.95, random_state = 0, stratify = y)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size = 0.8, random_state = 0, stratify = y_val)

n_activities = len(np.unique(y))

print(X_train.shape)#Prints train dataset size
print(X_val.shape)#Prints test dataset size

# A CNN only accepts 3 dimentional data so we are going to reshape() our data and just a dimension with a constant value.
X_train = X_train.reshape(X_train.shape + (1, ))
X_val = X_val.reshape(X_val.shape + (1, ))
X_test = X_test.reshape(X_test.shape + (1, ))

## Create the model
model = Sequential()
model.add(Conv2D(4, (3, 1), activation = 'relu', input_shape = (100, 3, 1))) 
model.add(Dropout(0.1))

model.add(Conv2D(4, (2, 2), activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(8, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(n_activities, activation='softmax'))

#Here we are compiling the model and fitting it to the training data. We will use 200 epochs to train the model.
#An epoch is an iteration over the entire data provided. validation_data is the data on which to evaluate the loss and any model metrics at the end of each epoch.
#The model will not be trained on this data. As metrics = ['accuracy'] the model will be evaluated based on the accuracy.

iterations = 15

lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=10000, decay_rate=0.9)
model.compile(optimizer=Adam(learning_rate = lr_schedule), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
history = model.fit(X_train, y_train, epochs = iterations, validation_data= (X_val, y_val), verbose=1)

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
array = ['Walking', 'Running']
mat = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_mat=mat, class_names= array, show_normed=True, figsize=(3,3))
plt.show()

print('X_test: ', X_train[0])
print('X_test shape: ', X_train[0].shape)
predictions = model.predict(X_train)
print(predictions)
