import DeepLearningModel
import glob

FREQUENCY = 200         # Frequency the microcontroller collected data
TARGET_FREQUENCY = 50   # Frequency to scale collected data to / Frequency of the TensorFlow model

model = DeepLearningModel.DeepLearningModel(FREQUENCY, TARGET_FREQUENCY)

# use glob.glob() to get all files that you want included in the dataset
# by using *.txt it gets all txt files in the selected directory
# using files.extend more files can be added to the selection
files = glob.glob("Datasets/walking/*.txt")
files.extend(glob.glob("Datasets/running/*.txt"))
files.extend(glob.glob("Datasets/stairs/*.txt"))
files.extend(glob.glob("Datasets/cycling/*.txt"))

# create_dataset loads all the data in the files for further use
model.create_dataset(files)

# save_dataset saves the created data in a file for faster loading when using the save datafiles used by create_dataset
model.save_dataset()

# ready_dataset splits all the data into train, test and validation data
model.ready_datasets()

# the three above function can be excecuted all at once with the create_save_and_ready_dataset function below
# model.create_save_and_ready_dataset()

