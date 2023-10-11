from typing import List
import numpy as np


from Possession import Possession

from Moment import Moment
from TemporalWindow import TemporalWindow


from globals import WINDOW_SIZE, MOMENT_SIZE

def createTemporalWindows(possessions: List[Possession]):
    for eachPossession in possessions:
        terminalActionIndex = eachPossession.terminalActionIndex
        temporalWindows = []

        if terminalActionIndex == -1:
            if len(eachPossession.moments) >= WINDOW_SIZE:
                temporalWindow = TemporalWindow()
                temporalWindow.moments = eachPossession.moments[-WINDOW_SIZE:]
                temporalWindow.label = 0


                temporalWindows.append(temporalWindow)
            else:
                temporalWindow = TemporalWindow()
                temporalWindow.moments = eachPossession.moments
                temporalWindow.label = 0
                temporalWindows.append(temporalWindow)
        else:
            numTemporalWindows = int((terminalActionIndex + 1) / WINDOW_SIZE)
            remainder = (terminalActionIndex + 1) % WINDOW_SIZE
            currentIndex = terminalActionIndex
            temporalWindowStartIndex = currentIndex - WINDOW_SIZE
            temporalWindowEndIndex = currentIndex
            for i in range(numTemporalWindows):
                temporalWindowStartIndex = currentIndex - WINDOW_SIZE
                temporalWindowEndIndex = currentIndex
                temporalWindow = TemporalWindow()
                temporalWindow.moments = eachPossession.moments[temporalWindowStartIndex + 1:temporalWindowEndIndex + 1]
                temporalWindow.label = eachPossession.moments[temporalWindowEndIndex].momentLabel
                currentIndex = currentIndex - WINDOW_SIZE
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
        


def processDataForLSTM(possessions : List[Possession]):

    inputTemporalWindowMatrix = []
    outputTemporalWindowLabels = []

    for eachPossession in possessions:

        for eachTemporalWindow in eachPossession.temporalWindows:

            momentObjectsOfTemporalWindow : List[Moment] = eachTemporalWindow.moments

            momentDataOfTemporalWindow = []
            labelDataOfTemporalWindow = eachTemporalWindow.label

            for eachMoment in momentObjectsOfTemporalWindow:

                if(len(eachMoment.momentArray) == MOMENT_SIZE):

                    momentDataOfTemporalWindow.append(eachMoment.momentArray)

            if(len(momentDataOfTemporalWindow) == WINDOW_SIZE):
            
                inputTemporalWindowMatrix.append(momentDataOfTemporalWindow)
                outputTemporalWindowLabels.append(labelDataOfTemporalWindow)

            


    return inputTemporalWindowMatrix, outputTemporalWindowLabels

