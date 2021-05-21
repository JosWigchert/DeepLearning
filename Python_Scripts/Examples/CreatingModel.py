import DeepLearningModel
import glob

FREQUENCY = 200                 # Frequency the microcontroller collected data
TARGET_FREQUENCY = 50           # Frequency to scale collected data to / Frequency of the TensorFlow model

modelName = 'TensorFlowModel'   # The name of the model used for saving / loading
epochs = 20                     # Number of cycles the model should run

model = DeepLearningModel.DeepLearningModel(FREQUENCY, TARGET_FREQUENCY)

# first load the dataset for learning
model.load_and_ready_datasets()

# create_model creates a TensorFlow model
model.create_model()

# train trains the model with the loaded datasets, with epochs as the number of cycles
model.train(epochs)

# save saves the model to a folder named with modelName
model.save_model(modelName)

# the three above function can be excecuted all at once with the create_train_and_save_model function below
# model.create_train_and_save_model(epochs, modelName)

# when a model is trained the show_plots function will show 3 plots that show the accuracy, the loss and how well the model performed with a confusion matrix
model.show_plots()