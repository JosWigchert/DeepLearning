import DeepLearningModel
import glob

FREQUENCY = 200                 # Frequency the microcontroller collected data
TARGET_FREQUENCY = 50           # Frequency to scale collected data to / Frequency of the TensorFlow model

modelName = 'TensorFlowModel'   # The name of the model used for saving / loading
epochs = 30                     # Number of cycles the model should run

model = DeepLearningModel.DeepLearningModel(FREQUENCY, TARGET_FREQUENCY)

# load_model loads a saved model created in the CreatingModel example
model.load_model(modelName)

# compile_tflite_model converts the model to a TensorFlow Lite model and converts that into a c header file that can be used in the microcontroller
model.compile_tflite_model(modelName)