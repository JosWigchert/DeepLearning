import DeepLearningModel
import glob

FREQUENCY = 200         # Frequency the microcontroller collected data
TARGET_FREQUENCY = 50   # Frequency to scale collected data to / Frequency of the TensorFlow model

model = DeepLearningModel.DeepLearningModel(FREQUENCY, TARGET_FREQUENCY)

# load_dataset loads the saved dataset created in the CreatingDataset example
model.load_dataset()

# ready_dataset splits all the data into train, test and validation data
model.ready_datasets()

# the two above function can be excecuted all at once with the load_and_ready_datasets function below
# this will be used by the CreatingModel example
# model.load_and_ready_datasets()


