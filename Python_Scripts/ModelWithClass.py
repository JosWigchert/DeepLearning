import DeepLearningModel
import glob
FREQUENCY = 200
TARGET_FREQUENCY = 50
modelName = 'Walking_Running_Stairs_Cycling_Model'
epochs = 100

model = DeepLearningModel.DeepLearningModel(FREQUENCY, TARGET_FREQUENCY)

files = glob.glob("Datasets/walking/*.txt")
files.extend(glob.glob("Datasets/running/*.txt"))
files.extend(glob.glob("Datasets/stairs/*.txt"))
files.extend(glob.glob("Datasets/cycling/*.txt"))

# model.create_save_and_ready_dataset(files)
# model.load_and_ready_datasets()
# model.create_train_and_save_model(epochs, modelName)
model.load_model(modelName)
model.compile_tflite_model(modelName)
model.show_plots()
# print(model.predict(np.array([np.ones(300).reshape(100, 3, 1), np.zeros(300).reshape(100, 3, 1)])))
