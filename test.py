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

@print_error_and_continue
def plotLoss(history):
    # Plot training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot Over Epochs')
    plt.legend()
    plt.savefig('Graphs/loss_plot.png')
    plt.show()

@print_error_and_continue
def plotAccuracy(history):
    # Plot training and validation accuracy
    plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Plot Over Epochs')
    plt.legend()
    plt.savefig('Graphs/accuracy_plot.png')
    plt.show()

@print_error_and_continue
def plotLearningCurve(history):
    train_loss = []  # To store training loss
    val_loss = []    # To store validation loss
    train_acc = []   # To store training accuracy
    val_acc = []     # To store validation accuracy

    train_loss.extend(history.history['loss'])
    val_loss.extend(history.history['val_loss'])
    train_acc.extend(history.history['categorical_accuracy'])
    val_acc.extend(history.history['val_categorical_accuracy'])

    num_examples_per_epoch = np.arange(1, len(train_loss) + 1) * len(X_train)

    # Plot learning curves for loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(num_examples_per_epoch, train_loss, label='Training Loss')
    plt.plot(num_examples_per_epoch, val_loss, label='Validation Loss')
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Learning Curve - Loss')

    # Plot learning curves for accuracy
    plt.subplot(1, 2, 2)
    plt.plot(num_examples_per_epoch, train_acc, label='Training Accuracy')
    plt.plot(num_examples_per_epoch, val_acc, label='Validation Accuracy')
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Learning Curve - Accuracy')

    plt.tight_layout()
    plt.savefig('Graphs/learning_curves.png')
    plt.show()


mapping = {
    0.0: "null",
    1.0: "FG Try",
    2.0: "Shoot F.",
    3.0: "Nonshoot F.",
    4.0: "Turnover"
}

@print_error_and_continue
def count_label_frequency(actual_time_series, predicted_time_series):
    unique_labels = np.unique(actual_time_series)
    label_counts_actual = {label: np.sum(actual_time_series == label) for label in unique_labels}
    label_counts_predicted = {label: np.sum(predicted_time_series == label) for label in unique_labels}
    return label_counts_actual, label_counts_predicted

@print_error_and_continue
def plot_label_frequency(label_counts_actual, label_counts_predicted):
    labels = [mapping[label] for label in label_counts_actual.keys()]
    actual_counts = list(label_counts_actual.values())
    predicted_counts = list(label_counts_predicted.values())

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, actual_counts, width, label='Actual', color='blue')
    rects2 = ax.bar(x + width/2, predicted_counts, width, label='Predicted', color='red')

    ax.set_xlabel('Labels')
    ax.set_ylabel('Frequency')
    ax.set_title('Label Frequency: Actual vs. Predicted')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.savefig('Graphs/label_frequency.png')
    plt.show()

@print_error_and_continue
def plot_percent_error(label_counts_actual, label_counts_predicted):
    labels = [mapping[label] for label in label_counts_actual.keys()]
    actual_counts = list(label_counts_actual.values())
    predicted_counts = list(label_counts_predicted.values())

    percent_errors = [(abs(actual - predicted) / actual) * 100 if actual != 0 else 0 for actual, predicted in zip(actual_counts, predicted_counts)]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, percent_errors, color='green')

    ax.set_xlabel('Labels')
    ax.set_ylabel('Percent Error')
    ax.set_title('Percent Error: Actual vs. Predicted')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    plt.savefig('Graphs/percent_error.png')
    plt.show()

@print_error_and_continue
def plotTimeSeries(history, X_test, y_test, model):
    y_pred = model.predict(X_test)
    actual_time_series = y_test
    predicted_time_series = y_pred

    actual_time_series = np.array(actual_time_series)
    predicted_time_series = np.array(predicted_time_series)

    # Extract the selected index with the highest probability for each time step
    selected_indices = [np.argmax(probabilities) for probabilities in predicted_time_series]

    # Plot actual vs. predicted time series
    plt.figure(figsize=(10, 6))
    plt.plot(actual_time_series, label='Actual Time Series', color='blue', marker='o')
    plt.plot(selected_indices, label='Predicted Time Series', color='red', marker='x', linestyle='--')
    plt.xlabel('Time Steps')
    plt.ylabel('Selected Index (0-4)')
    plt.legend()
    plt.title('Actual vs. Predicted Time Series')
    plt.savefig('Graphs/time_series.png')
    plt.show()

    # Calculate and plot label frequency and percent error
    label_counts_actual, label_counts_predicted = count_label_frequency(actual_time_series, selected_indices)
    plot_label_frequency(label_counts_actual, label_counts_predicted)
    plot_percent_error(label_counts_actual, label_counts_predicted)



if __name__ == "__main__":
    # Specify the range of games you want to train on
    startGameNumber = 20
    endGameNumber = 300
    batch_size = 5  # Number of games to process in each batch

    histories = []  # To store training histories for each batch

    for i in range(startGameNumber, endGameNumber + 1, batch_size):
        currentStartGameNumber = i
        currentEndGameNumber = min(i + batch_size - 1, endGameNumber)

        # Extract game JSON data for the current batch
        extractFilesToDestinationFolder(currentStartGameNumber, currentEndGameNumber)

        datasetDirectoryVariable = os.listdir(destination_folder)

        # Get Input/Output Data
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
        model_directory = r"model2"
        model2 = createModel()

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

        with open(f'training_history_batch_{i}.pkl', 'wb') as file:
            pickle.dump(history.history, file)


        # Delete extracted JSON files for the current batch
        delete_files_in_folder(destination_folder)

    # Generate plots for the entire training process
    plotAccuracy(histories)
    plotLoss(histories)
    plotLearningCurve(histories)
    plotTimeSeries(histories)
    
    print("DONE")
