import os
import py7zr
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from Possession import Possession
from typing import List
from helperFunctions import createTemporalWindows, processDataForLSTM
from getPossessionsFromJSON import MomentPreprocessingClass
from globals import WINDOW_SIZE, MOMENT_SIZE, print_error_and_continue
import pickle
import os
import py7zr

from globals import WINDOW_SIZE, MOMENT_SIZE, print_error_and_continue

folder_path_with_7z = r"C:\Users\rayya\Desktop\NBA-Player-Movements\data\2016.NBA.Raw.SportVU.Game.Logs"
destination_folder = r"Current_Training_JSON"

@print_error_and_continue
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


@print_error_and_continue
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

from typing import List
from helperFunctions import createTemporalWindows, processDataForLSTM
import numpy as np
from getPossessionsFromJSON import MomentPreprocessingClass

@print_error_and_continue
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
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau  # save model and learning rate annealing
from keras.losses import CategoricalCrossentropy # For loss function
from keras.metrics import CategoricalAccuracy
from keras.optimizers import Adam

@print_error_and_continue
def createModel() -> Sequential:

    model2 = Sequential()
    model2.add(InputLayer((WINDOW_SIZE, MOMENT_SIZE)))
    model2.add(LSTM(64))
    model2.add(Dense(8, activation='relu'))
    model2.add(Dense(8, activation='sigmoid')) 
    model2.add(Dropout(0.5))  # prevents overfitting
    model2.add(Dense(16, activation='relu'))  
    model2.add(Dense(5, activation='softmax'))
    model2.summary()

    return model2




if __name__ == "__main__":
    # Specify the range of games you want to train on
    startGameNumber = 3
    endGameNumber = 20
    batch_size = 5  # Number of games to process in each batch

    histories = []  # To store training histories for each batch

    for i in range(startGameNumber, endGameNumber + 1, batch_size):
        currentStartGameNumber = i
        currentEndGameNumber = min(i + batch_size - 1, endGameNumber)

        # Extract game JSON data for the current batch
        extractFilesToDestinationFolder(currentStartGameNumber, currentEndGameNumber)

        datasetDirectoryVariable = os.listdir(destination_folder)

        # Get Input/Output Data
        print(f"Starting training on Games {currentStartGameNumber} to {currentEndGameNumber}")
        inputMatrix, outputVector = getInputOutputData(datasetDirectoryVariable)

        inputMatrix = np.array(inputMatrix)   # SHAPE: number of windows 1500, WINDOW_SIZE, MOMENT_SIZE
        outputVector = np.array(outputVector) # SHAPE: number of windows 1500, 1 

        print(inputMatrix.shape)
        print(outputVector.shape)

        # Organize Train/Test/Validation data
        X_train, X_rem, y_train, y_rem = train_test_split(inputMatrix, outputVector, train_size=0.8, random_state=42)
        X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

        y_train_encoded = to_categorical(y_train, num_classes=5)
        y_valid_encoded = to_categorical(y_valid, num_classes=5)

        # Create model
        from keras.models import load_model

        model_directory = r"model2"
        model2 : Sequential

        if not os.path.exists(model_directory):
            model2 = createModel()
        else:
            model2 = load_model(model_directory)

        # Train the model
        cp = ModelCheckpoint(f"{model_directory}", save_best_only=True)
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001
        )

        model2.compile(
            loss=CategoricalCrossentropy(),
            optimizer=Adam(learning_rate=0.001),
            metrics=[CategoricalAccuracy()]
        )

        history = model2.fit(
            X_train,
            y_train_encoded,
            validation_data=(X_valid, y_valid_encoded),
            epochs=50,
            callbacks=[cp, reduce_lr],
            batch_size=8
        )

        training_data_for_pickle = {
            'history' : history.history,
            'X_train' : X_train,
            'X_test'  : X_test,
            'y_test'  : y_test
        }

        with open(f'training_history_batches/{i}.pkl', 'wb') as file:
            pickle.dump(training_data_for_pickle, file)


        # Delete extracted JSON files for the current batch
        delete_files_in_folder(destination_folder)

    print("DONE")
