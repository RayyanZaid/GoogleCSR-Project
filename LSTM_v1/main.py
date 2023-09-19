# 1) Read JSON and preprocess the data into a CSV

# 2) Read from CSV and create possessions

# 3) Go through possessions and create Temporal Windows

# 4) Process data so that it's in a input matrix : output vector format
    # ex.
    # Temporal Window Moments       :   Temporal Window Labels    
    #       [                           [
    #           [[25D], [25D]],             [0],
    #           [[25D], [25D]],             [1],
    #       ]                           ]

    # INPUT:
    # Each element is a Temporal Window that consists of several moments
    # Each Moment consists of 25 pieces of data

    # OUTPUT:
    # Each element is the output label corresponding to the input temporal window


# 5) Input each temporal window into the LSTM

from typing import List
import numpy as np

from MomentPreprocessing.MomentPreprocessingMain import MomentPreprocessingClass
from Possession import Possession
from getPossessionsFromCSV import getData
from Moment import Moment
from TemporalWindow import TemporalWindow
# 1) Read JSON and preprocess the data into a CSV

# Already preprocessed into CSV

# obj = MomentPreprocessingClass(r"C:\Users\rayya\OneDrive\Desktop\GoogleCSR-Project\Datasets\0021500524.json")
# obj.read_json()
# obj.iterateThroughEvents()

# 2) Read from CSV and create possessions

possessions : List[Possession] = getData()


# 3) Go through possessions and create Temporal Windows


T = 128 

def createTemporalWindows(possessions: List[Possession]):
    for eachPossession in possessions:
        terminalActionIndex = eachPossession.terminalActionIndex
        temporalWindows = []

        if terminalActionIndex == -1:
            if len(eachPossession.moments) >= T:
                temporalWindow = TemporalWindow()
                temporalWindow.moments = eachPossession.moments[-T:]
                temporalWindow.label = 0
                temporalWindows.append(temporalWindow)
            else:
                temporalWindow = TemporalWindow()
                temporalWindow.moments = eachPossession.moments
                temporalWindow.label = 0
                temporalWindows.append(temporalWindow)
        else:
            numTemporalWindows = int((terminalActionIndex + 1) / T)
            remainder = (terminalActionIndex + 1) % T
            currentIndex = terminalActionIndex

            for i in range(numTemporalWindows):
                temporalWindowStartIndex = currentIndex - T
                temporalWindowEndIndex = currentIndex
                temporalWindow = TemporalWindow()
                temporalWindow.moments = eachPossession.moments[temporalWindowStartIndex + 1:temporalWindowEndIndex + 1]
                temporalWindow.label = eachPossession.moments[temporalWindowEndIndex].momentLabel
                currentIndex = currentIndex - T
                temporalWindows.insert(0, temporalWindow)

            temporalWindow = TemporalWindow()
            temporalWindow.moments = eachPossession.moments[temporalWindowStartIndex: temporalWindowStartIndex+remainder]
            temporalWindow.label = eachPossession.moments[temporalWindowStartIndex].momentLabel
            temporalWindows.insert(0, temporalWindow)

        eachPossession.temporalWindows = temporalWindows
        print("Added Temporal Windows of Possession with Terminal Action")

# Now call the function
createTemporalWindows(possessions)


# 4) Process data so that it's in a input matrix : output vector format


# This function:
    # Input : List of Possessions with Temporal Windows updated
    # Ouput : the input matrix and output vector

def processDataForLSTM(possessions : List[Possession]):

    inputTemporalWindowMatrix = []
    outputTemporalWindowLabels = []

    for eachPossession in possessions:

        for eachTemporalWindow in eachPossession.temporalWindows:

            momentObjectsOfTemporalWindow : List[Moment] = eachTemporalWindow.moments

            momentDataOfTemporalWindow = []
            labelDataOfTemporalWindow = [eachTemporalWindow.label]

            for eachMoment in momentObjectsOfTemporalWindow:
                momentDataOfTemporalWindow.append(eachMoment.momentArray)

            
            inputTemporalWindowMatrix.append(momentDataOfTemporalWindow)
            outputTemporalWindowLabels.append(labelDataOfTemporalWindow)

            


    return inputTemporalWindowMatrix, outputTemporalWindowLabels

inputMatrix, outputVector = processDataForLSTM(possessions)

# inputMatrix = np.array(inputMatrix)
outputVector = np.array(outputVector)

# print(inputMatrix.shape)
print(outputVector.shape)