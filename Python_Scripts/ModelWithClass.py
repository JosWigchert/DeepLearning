import DeepLearningModel
import numpy as np
import glob

FREQUENCY = 200
TARGET_FREQUENCY = 50
modelName = 'Walking_Running_Model'
epochs = 50

model = DeepLearningModel.DeepLearningModel(FREQUENCY, TARGET_FREQUENCY)

files = glob.glob("Datasets/walking/*.txt")
files.extend(glob.glob("Datasets/running/*.txt"))

model.create_dataset(files)
model.save_dataset()
model.ready_datasets()
model.create_train_and_save_model(epochs, modelName, plot=True)
model.confusion_matrix()
# model.create_save_and_ready_dataset()
#model.load_and_ready_datasets()

#model.load_model(modelName)
#model.compile_tflite_model(modelName)
#print(model.predict(np.array([model.X_test[0]])))