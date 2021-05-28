import DeepLearningModel
import glob
import numpy as np

FREQUENCY = 200                 # Frequency the microcontroller collected data
TARGET_FREQUENCY = 50           # Frequency to scale collected data to / Frequency of the TensorFlow model

modelName = 'TensorFlowModel'   # The name of the model used for saving / loading
epochs = 30                     # Number of cycles the model should run

model = DeepLearningModel.DeepLearningModel(FREQUENCY, TARGET_FREQUENCY)

# load_model loads a saved model created in the CreatingModel example
model.load_model(modelName)

# predict, predicts the output of the array as input
# the function expects an array of dataframes (a dataframe is an array of 100 x, y, z values. shaped as (100, 3, 1))

# using 1 dataframe
predictOutput = model.predict(np.array([np.ones(300).reshape(100, 3, 1)]))
print('Predict output of 1 dataframe')
print(predictOutput)

# using multiple dataframes
predictOutput = model.predict(np.array([np.ones(300).reshape(100, 3, 1) * -2, np.ones(300).reshape(100, 3, 1) * -1, np.zeros(300).reshape(100, 3, 1), np.ones(300).reshape(100, 3, 1), np.ones(300).reshape(100, 3, 1) * 2]))
print('Predict output of multiple dataframes')
print(predictOutput)