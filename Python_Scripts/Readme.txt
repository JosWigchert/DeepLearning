Modelconversion.py: Allows you to convert a normal tensorflow model to a TFlite model(optimized for size) and converts it into a C header file.
Maximize_dataset.py: Converts input .csv data from 200Hz to 50Hz measurements and writes them to new a .csv file (It would be desirable to make this script more universal in future updates)
TrainWalking_and_Running.py: Script that takes input .csv(with walking and running data) and trains a model that it saves to your current directory, various plots will be shown to indicate the performance of the model.
balanced_data_50hz.csv: Already balanced data set that is "downsampled" to 50Hz and can be used with the TrainWalking_and_Running.py script
