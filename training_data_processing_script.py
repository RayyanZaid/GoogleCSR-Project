import os
import py7zr
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from Possession import Possession
from typing import List
from helperFunctions import createTemporalWindows, processDataForLSTM
from getPossessionsFromJSON import MomentPreprocessingClass
from globals import print_error_and_continue
import pickle
from nba_api.stats.endpoints import playbyplay
import re

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
    counter = 0

    sevenz_files = [filename for filename in os.listdir(folder_path_with_7z) if filename.endswith('.7z')][inputStartIndex-1:inputEndIndex]

    if not sevenz_files:
        exit()

    while counter < len(sevenz_files):

        startIndex = counter
        endIndex = startIndex + grouping_size

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




def getTotalPointsScoredInAGame(jsonPath):
    
    pattern = r'\d+'

    match = re.search(pattern, jsonPath)
        
    gameID = match.group()

    
    plays = playbyplay.PlayByPlay(gameID).get_normalized_dict()['PlayByPlay']
    home_score = 0
    visitor_score = 0

    for play in plays:
        if play['EVENTMSGTYPE'] == 13:  # Check for end of 4th quarter

            score_text = play['SCORE']
            if score_text:
                # Split the score string into home and visitor scores
                home, visitor = score_text.split('-')
                home_score = int(home.strip())
                visitor_score = int(visitor.strip())

    return home_score + visitor_score


@print_error_and_continue
def getInputOutputData(datasetDirectoryVariable):

    allScore = 0
    allPossessions : List[Possession] = []
    for eachJSON in datasetDirectoryVariable:
        json_path = os.path.join(destination_folder, eachJSON)
        print(json_path)
        
        momentPreprocessing : MomentPreprocessingClass = MomentPreprocessingClass(json_path)
        possessions : List[Possession] = momentPreprocessing.getData(json_path)
        createTemporalWindows(possessions)
        allPossessions.extend(possessions)
        score = getTotalPointsScoredInAGame(json_path)
        allScore += score



    inputMatrix , outputVector = processDataForLSTM(possessions)
    

    return inputMatrix, outputVector, len(allPossessions), allScore

if __name__ == "__main__":

    # Specify the range of games to train
    startGameNumber = 416
    endGameNumber = 500
    grouping_size = 5  # Number of games to process in each group


    totalScore = 0
    totalNumPossessions = 0
    results_data = []

    for i in range(startGameNumber, endGameNumber + 1, grouping_size):
        currentStartGameNumber = i
        currentEndGameNumber = min(i + grouping_size - 1, endGameNumber)

        # Extract game JSON data for the current group
        extractFilesToDestinationFolder(currentStartGameNumber, currentEndGameNumber)

        datasetDirectoryVariable = os.listdir(destination_folder)

        # Get Input/Output Data
        print(f"Starting training on Games {currentStartGameNumber} to {currentEndGameNumber}")
        inputMatrix, outputVector, numPossessions, score = getInputOutputData(datasetDirectoryVariable)

        if len(inputMatrix) == 0:   # if JSON file is corrupted
            delete_files_in_folder(destination_folder)
            continue

        totalScore += score
        totalNumPossessions += numPossessions

        inputMatrix = np.array(inputMatrix)   # SHAPE: number of windows 1500, WINDOW_SIZE, MOMENT_SIZE
        outputVector = np.array(outputVector) # SHAPE: number of windows 1500, 1 

        # print(inputMatrix.shape)
        # print(outputVector.shape)

        

        # Organize Train/Test/Validation data
        X_train, X_rem, y_train, y_rem = train_test_split(inputMatrix, outputVector, train_size=0.8, random_state=42)
        X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

        y_train_encoded = to_categorical(y_train, num_classes=5)
        y_valid_encoded = to_categorical(y_valid, num_classes=5)

        training_data_for_pickle = {
            # 'history' : history.history,
            'X_train' : X_train,
            'X_test'  : X_test,
            'y_train_encoded' : y_train_encoded,
            'y_test'  : y_test,
            'X_valid' : X_valid,
            'y_valid_encoded' : y_valid_encoded
            
        }

        with open(f'training_history_groups/{i}.pkl', 'wb') as file:
            pickle.dump(training_data_for_pickle, file)


        results_data.append(
            {
                "Range": (currentStartGameNumber, currentEndGameNumber),
                "Score": totalScore,
                "Number of Possessions": totalNumPossessions,
            }
        )

        # Print results to the console
        for result in results_data:
            print(
                f"Range {result['Range']} : (Score: {result['Score']}), (Number of Possessions: {result['Number of Possessions']})"
            )

        # Save results to a text file
        with open(f"training_results{startGameNumber}.txt", "w") as txt_file:
            for result in results_data:
                txt_file.write(
                    f"Range {result['Range']} : (Score: {result['Score']}), (Number of Possessions: {result['Number of Possessions']})\n"
                )
        # Delete extracted JSON files for the current group
        delete_files_in_folder(destination_folder)

    print("DONE")
