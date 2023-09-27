from typing import List
import numpy as np

from MomentPreprocessing.MomentPreprocessingMain import MomentPreprocessingClass
from Possession import Possession
from getPossessionsFromJSON import getData
from Moment import Moment
from TemporalWindow import TemporalWindow


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
            temporalWindowStartIndex = currentIndex - T
            temporalWindowEndIndex = currentIndex
            for i in range(numTemporalWindows):
                temporalWindowStartIndex = currentIndex - T
                temporalWindowEndIndex = currentIndex
                temporalWindow = TemporalWindow()
                temporalWindow.moments = eachPossession.moments[temporalWindowStartIndex + 1:temporalWindowEndIndex + 1]
                temporalWindow.label = eachPossession.moments[temporalWindowEndIndex].momentLabel
                currentIndex = currentIndex - T
                temporalWindows.insert(0, temporalWindow)

            temporalWindow = TemporalWindow()

            if temporalWindowStartIndex < 0:
                temporalWindowStartIndex = 0

            if temporalWindowStartIndex >= len(eachPossession.moments):
                temporalWindowStartIndex = len(eachPossession.moments) - remainder
            temporalWindow.moments = eachPossession.moments[temporalWindowStartIndex: temporalWindowStartIndex+remainder]

            
            temporalWindow.label = eachPossession.moments[temporalWindowStartIndex].momentLabel
            temporalWindows.insert(0, temporalWindow)

        eachPossession.temporalWindows = temporalWindows
        print("Added Temporal Windows of Possession with Terminal Action")


def processDataForLSTM(possessions : List[Possession]):

    inputTemporalWindowMatrix = []
    outputTemporalWindowLabels = []

    for eachPossession in possessions:

        for eachTemporalWindow in eachPossession.temporalWindows:

            momentObjectsOfTemporalWindow : List[Moment] = eachTemporalWindow.moments

            momentDataOfTemporalWindow = []
            labelDataOfTemporalWindow = eachTemporalWindow.label

            for eachMoment in momentObjectsOfTemporalWindow:

                if(len(eachMoment.momentArray) == 25):

                    momentDataOfTemporalWindow.append(eachMoment.momentArray)

            if(len(momentDataOfTemporalWindow) == 128):
            
                inputTemporalWindowMatrix.append(momentDataOfTemporalWindow)
                outputTemporalWindowLabels.append(labelDataOfTemporalWindow)

            


    return inputTemporalWindowMatrix, outputTemporalWindowLabels



from keras.models import Sequential
from keras.layers import LSTM, Dense, Masking

# def loadAndTrainLSTM(inputData, labelData):

#     model = Sequential()

#     # Add a Masking layer to handle the masking automatically
#     model.add(Masking(mask_value=-1, input_shape=(128, 25)))  # Replace -1 with the appropriate mask value

#     # Add the LSTM layer with mask support
#     model.add(LSTM(units=64, activation='tanh', return_sequences=False, input_shape=(128, 25), mask_zero=True))  
#     # Set mask_zero=True to handle masking

#     # Add the output layer with sigmoid activation for binary classification
#     model.add(Dense(units=1, activation='sigmoid'))

#     # Compile the model
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.fit(inputData, labelData, epochs=10, batch_size=32)  
#     model.save("lstm_v1.h5")


from keras.models import load_model

def predict(testData):
    loaded_model = load_model("lstm_v1.h5")
    predictions = loaded_model.predict(testData)
    print()
