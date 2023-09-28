import os
import py7zr



folder_path_with_7z = r"D:\coding\NBA-Player-Movements\data\2016.NBA.Raw.SportVU.Game.Logs"
destination_folder = r"D:\coding\GoogleCSR-Project\Dataset"


def delete_files_in_folder(folder_path):
    try:
        # List all files in the folder
        files = os.listdir(folder_path)

        # Iterate through the files and delete them
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")

        print(f"All files in {folder_path} have been deleted.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")



def extractFilesToDestinationFolder(inputStartIndex, inputEndIndex):

    

    batch_size = 5  # Number of files to process in each batch
    counter = 0

    sevenz_files = [filename for filename in os.listdir(folder_path_with_7z) if filename.endswith('.7z')][inputStartIndex-1:inputEndIndex]

    if not sevenz_files:
        exit()

    while counter < len(sevenz_files):
        

        # Process files in batches
        startIndex = counter
        endIndex = startIndex + batch_size

        if endIndex >= len(sevenz_files):
            endIndex = len(sevenz_files)

        for filename in sevenz_files:
            file_path = os.path.join(folder_path_with_7z, filename)

            with py7zr.SevenZipFile(file_path, mode='r') as archive:
                archive.extractall(destination_folder)
                print(f"Extracted {filename} to {destination_folder}")

            counter += 1


        

        extracted_files = os.listdir(destination_folder)
        print(f"Extracted files in {destination_folder}: {', '.join(extracted_files)}")
        





        if endIndex == len(sevenz_files) - 1:
            break

    print(f"Processed {counter} .7z files in folder: {folder_path_with_7z}")



from Possession import Possession
from getPossessionsFromJSON import getData
from typing import List
from helperFunctions import createTemporalWindows, processDataForLSTM
import numpy as np
from getPossessionsFromJSON import MomentPreprocessingClass

def getInputOutputData(datasetDirectoryVariable):

    allPossessions : List[Possession] = []
    for eachJSON in datasetDirectoryVariable:
        json_path = os.path.join(destination_folder, eachJSON)
        print(json_path)
        
        momentPreprocessing : MomentPreprocessingClass = MomentPreprocessingClass(json_path)
        possessions : List[Possession] = momentPreprocessing.getData(json_path)
        createTemporalWindows(possessions)
        allPossessions.extend(possessions)

    inputMatrix , outputVector = processDataForLSTM(possessions)

    return inputMatrix, outputVector


from keras.models import Sequential # Sequential Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint # To save model
from keras.losses import CategoricalCrossentropy # For loss function
from keras.metrics import CategoricalAccuracy
from keras.optimizers import Adam

def createModel() -> Sequential:

    model1 = Sequential()
    model1.add(InputLayer((128, 25)))
    model1.add(LSTM(64))
    model1.add(Dense(8, activation='relu'))
    model1.add(Dense(8, activation='sigmoid')) 
    model1.add(Dropout(0.5))  # prevents overfitting
    model1.add(Dense(16, activation='relu'))  
    model1.add(Dense(5, activation='softmax'))
    model1.summary()

    return model1




#  Cell 1 -- Get Start & End Games

startGameNumber = int(input("What game number do you want to START the training?"))
endGameNumber = int(input("What game number do you want to END the training?"))

#  Cell 2 -- Extract game JSON data

step_size = 5
for i in range(startGameNumber,endGameNumber+1,step_size):    

    currentStartGameNumber = i
    currentEndGameNumber = currentStartGameNumber + step_size

    isDone = False

    if currentEndGameNumber > endGameNumber:
        currentEndGameNumber = endGameNumber
        isDone = True

    extractFilesToDestinationFolder(currentStartGameNumber,currentEndGameNumber) # start by extracting the JSON files into the dataset folder
        
    datasetDirectoryVariable = os.listdir(destination_folder)

    #  Cell 3 -- Get Input/Output Data

    inputMatrix , outputVector = getInputOutputData(datasetDirectoryVariable)

    inputMatrix = np.array(inputMatrix)   # SHAPE: number of windows 1500, 128, 25
    outputVector = np.array(outputVector) # SHAPE: number of windows 1500, 1 

    print(inputMatrix.shape)
    print(outputVector.shape)


    #  Cell 4 -- Organize Train/Test/Validation data

    from sklearn.model_selection import train_test_split
    from keras.utils import to_categorical

    X_train, X_rem, y_train, y_rem = train_test_split(inputMatrix, outputVector, train_size=0.8, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

    y_train_encoded = to_categorical(y_train, num_classes=5)
    y_valid_encoded = to_categorical(y_valid, num_classes=5)


    #  Cell 5 -- Create model

    from keras.models import load_model

    model_directory = r"D:\coding\GoogleCSR-Project\model1"
    model1 : Sequential

    if not os.path.exists(model_directory):
        model1 = createModel()
    else:
        model1 = load_model(model_directory)


    #  Cell 6 -- Train

    cp = ModelCheckpoint(r"D:\coding\GoogleCSR-Project\model1", save_best_only=True) # saves model with lowest validation loss
    model1.compile(loss=CategoricalCrossentropy(), optimizer=Adam(learning_rate=0.01), metrics=[CategoricalAccuracy()]) # higher the learning rate, the faster the model will try to decrease the loss function
    model1.fit(X_train, y_train_encoded, validation_data=(X_valid, y_valid_encoded), epochs=10, callbacks=[cp], batch_size=8)

    # Cell 7 -- Delete

    delete_files_in_folder(destination_folder) # end by deleting the JSON files from the dataset folder

    if isDone:
         break





# NOTES
# Trained : 1 -- 20
# Next game to train : 

