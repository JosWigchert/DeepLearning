import pandas as pd
import scipy.stats as stats
import numpy as np
import io
import matplotlib.pyplot as plt 
import glob
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam

from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

import pickle

class DeepLearningModel:
    def __init__(self, FREQUENCY, TARGET_FREQUENCY):
        print(tf.version)

        self.FREQUENCY = FREQUENCY
        self.TARGET_FREQUENCY = TARGET_FREQUENCY
        self.DOWNSAMPLE = int(FREQUENCY/TARGET_FREQUENCY)
        print(self.DOWNSAMPLE)

        self.scale = [1024, 1024, 1024]
        self.mean = [0, 0, 0]

        self.Fs = self.TARGET_FREQUENCY #Samples per second
        self.Seconds = 2 # Seconds the frame covers
        self.frame_size = self.Fs*self.Seconds
        self.hop_size = 10 # The movement of the frame this results in overlapping data

    def create_save_and_ready_dataset(self):
        self.create_dataset()
        self.save_dataset()
        self.ready_datasets()

    def create_dataset(self):
        files = glob.glob("Datasets/walking/*.txt")
        files.extend(glob.glob("Datasets/running/*.txt"))
        files.extend(glob.glob("Datasets/stairs/*.txt"))
        files.extend(glob.glob("Datasets/cycling/*.txt"))

        self.datasets = self.preprocess_files(files)

        self.X, self.Y = self.create_frames(self.datasets[0], self.frame_size, self.hop_size)

        for i in range(1,len(self.datasets)):
            temp_x, temp_y = self.create_frames(self.datasets[i], self.frame_size, self.hop_size)
            self.X = np.append(self.X, temp_x, axis=0)
            self.Y = np.append(self.Y, temp_y)

    def save_dataset(self):
        np.save('serial_dataset_x.npy', self.X)

        np.save('serial_dataset_y.npy', self.Y)

        print('shapes: ', self.X.shape, self.Y.shape)   

    def load_dataset(self):
        self.Y = np.load('serial_dataset_y.npy')

        self.X = np.load('serial_dataset_x.npy')        

        print('X and Y loaded from files, shapes: ', self.X.shape, self.Y.shape)

    def load_and_ready_datasets(self):
        self.load_dataset()
        self.ready_datasets()

    def ready_datasets(self):
        # Here we are dividing the data into training data and test data using train_test_split() from sklearn 
        # which we have already imported. We are going to use 80% of the data for training the model and 20% of the data for testing. 
        # random_state controls the shuffling applied to the data before applying the split. stratify = y splits the data in a stratified fashion, using y as the class labels.
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X, self.Y, test_size = 0.8, random_state = 0, stratify = self.Y)
        self.X_val, self.X_test, self.Y_val, self.Y_test = train_test_split(self.X_val, self.Y_val, test_size = 0.5, random_state = 0, stratify = self.Y_val)

        print(self.X_train.shape)#Prints train dataset size
        print(self.X_val.shape)#Prints test dataset size

        # A CNN only accepts 3 dimentional data so we are going to reshape() our data and just a dimension with a constant value.
        self.X_train = self.X_train.reshape(self.X_train.shape + (1, ))
        self.X_val = self.X_val.reshape(self.X_val.shape + (1, ))
        self.X_test = self.X_test.reshape(self.X_test.shape + (1, ))

    def create_model(self):
        n_activities = len(np.unique(self.Y))

        ## Create the model
        self.model = Sequential()
        self.model.add(Conv2D(4, (3, 1), activation = 'relu', input_shape = (100, 3, 1))) 
        self.model.add(Dropout(0.1))

        self.model.add(Conv2D(4, (2, 2), activation='relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Flatten())
        self.model.add(Dense(8, activation = 'relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(4, activation='softmax'))

    def train(self, epochs):
        #Here we are compiling the model and fitting it to the training data. We will use 200 epochs to train the model.
        #An epoch is an iteration over the entire data provided. validation_data is the data on which to evaluate the loss and any model metrics at the end of each epoch.
        #The model will not be trained on this data. As metrics = ['accuracy'] the model will be evaluated based on the accuracy.

        self.epochs = epochs

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=10000, decay_rate=0.9)
        self.model.compile(optimizer=Adam(learning_rate = lr_schedule), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
        #model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
        self.history = self.model.fit(self.X_train, self.Y_train, epochs = self.epochs, validation_data=(self.X_val, self.Y_val), verbose=1)

    def save_model(self, name: str):
        self.model.save(name)

    def load_model(self, name: str):
        self.model = tf.keras.models.load_model(name)

    def compile_tflite_model(self, name: str):
        MODELS_DIR = name+ '/'
        if not os.path.exists(MODELS_DIR):
            os.mkdir(MODELS_DIR)
        MODEL_TF = 'name'
        MODEL_NO_QUANT_TFLITE = MODELS_DIR + name + '_no_quant.tflite'
        MODEL_TFLITE = MODELS_DIR + name + '.tflite'
        MODEL_TFLITE_MICRO = MODELS_DIR + name + '.cc'

        # Convert the model to the TensorFlow Lite format without quantization
        converter = tf.lite.TFLiteConverter.from_saved_model(name)
        # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        model_no_quant_tflite = converter.convert()

        # Save the model to disk
        open(MODEL_NO_QUANT_TFLITE, "wb").write(model_no_quant_tflite)

        #size_tf = os.path.getsize(MODEL_TF)
        #print(size_tf)
        size_no_quant_tflite = os.path.getsize(MODEL_NO_QUANT_TFLITE)
        print('Size of file ' + str(size_no_quant_tflite) + ' bytes')

        c_model_name = name
        with open(c_model_name + '.h', 'w') as file:
            file.write(self.hex_to_c_array(model_no_quant_tflite, c_model_name))

    def hex_to_c_array(self, hex_data, var_name):
        c_str = ''

        # Create header guard
        c_str += '#ifndef ' + var_name.upper() + '_H\n'
        c_str += '#define ' + var_name.upper() + '_H\n\n'

        # Add array length at top of file
        c_str += '\nunsigned int ' + var_name + '_len = ' + str(len(hex_data)) + ';\n'

        # Declare C variable
        c_str += 'unsigned char ' + var_name + '[] = {'
        hex_array = []
        for i, val in enumerate(hex_data) :

            # Construct string from hex
            hex_str = format(val, '#04x')

            # Add formatting so each line stays within 80 characters
            if (i + 1) < len(hex_data):
                hex_str += ','
            if (i + 1) % 12 == 0:
                hex_str += '\n '
            hex_array.append(hex_str)

        # Add closing brace
        c_str += '\n ' + format(' '.join(hex_array)) + '\n};\n\n'

        # Close out header guard
        c_str += '#endif //' + var_name.upper() + '_H'

        return c_str

    def create_train_and_save_model(self, epochs, name: str):
        self.create_model()
        self.train(epochs)
        self.save_model(name)

    def show_plots(self):
        # Plot training & validation accuracy values
        epoch_range = range(1, self.epochs+1)
        plt.plot(epoch_range, self.history.history['accuracy'])
        plt.plot(epoch_range, self.history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(epoch_range, self.history.history['loss'])
        plt.plot(epoch_range, self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()

        Y_pred = self.model.predict_classes(self.X_test)
        names = ['Walking', 'Running', 'Cycling', 'Climbing Stairs']
        uniqueTest = np.unique(self.Y_test)
        uniquePred = np.unique(Y_pred)
        maxTypes = max(len(uniqueTest), len(uniquePred))
        mat = confusion_matrix(self.Y_test, Y_pred)
        plot_confusion_matrix(conf_mat=mat, class_names=names[0:maxTypes], show_normed=True, figsize=(maxTypes,maxTypes))
        plt.show()

    def predict(self, inp):
        print('predicting with shape', inp.shape)
        return self.model.predict(inp)

    def preprocess_files(self, filenames):
        datasets = []

        for f in filenames:
            datasets.extend(self.preprocess_file(f))

        return datasets


    def preprocess_file(self, filename):
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
                if (len(split) >= 4) and i > self.FREQUENCY*10:
                    x.append((int(split[0]) - self.mean[0])/self.scale[0])
                    y.append((int(split[1]) - self.mean[1])/self.scale[1])
                    z.append((int(split[2]) - self.mean[2])/self.scale[2])
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
                    if len(df)/self.FREQUENCY > 5:
                        for j in range(self.DOWNSAMPLE):
                            downsampled = df.iloc[j::self.DOWNSAMPLE, :]
                            datasets.append(downsampled)
            except:
                print('error')
        f.close()

        return datasets

    def create_frames(self, df, frame_size, hop_size):
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