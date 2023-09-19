# 1) Read JSON and preprocess the data into a CSV

# 2) Read from CSV and create possessions

# 3) Go through possessions and create Temporal Windows

# 4) Input each temporal window into the LSTM


from MomentPreprocessing.MomentPreprocessingMain import MomentPreprocessingClass
from Possession import Possession
from getPossessionsFromCSV import getData
from Moment import Moment
from typing import List
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

def createTemporalWindows(possessions : List[Possession]):
    
    temporalWindows : List[TemporalWindow] = []


    for eachPossession in possessions:

        terminalActionIndex = eachPossession.terminalActionIndex

        # null terminal action

        temporalWindow : List[Moment] = []

        if terminalActionIndex == -1:
            # collect the last "T" moments of the possesion
            # that will be the only temporal window from this possession

            
            if len(eachPossession.moments) >= 128:
                temporalWindow = eachPossession.moments[-128:]
            else:
                temporalWindow = eachPossession.moments

            eachPossession.temporalWindows.append(temporalWindow)

            print("Added Temporal Window of Possesion with NO Terminal Action")

        else:
            numTemporalWindows = int((terminalActionIndex+1) / T)
            remainder = (terminalActionIndex+1) % T

            endIndex = terminalActionIndex + 1 - (numTemporalWindows * T + remainder)

            currentIndex = terminalActionIndex

            for i in range(numTemporalWindows):
                temporalWindowStartIndex = currentIndex - T
                temporalWindowEndIndex = currentIndex
                temporalWindow = eachPossession.moments[temporalWindowStartIndex+1:temporalWindowEndIndex+1]
                currentIndex = currentIndex - T
                eachPossession.temporalWindows.insert(0,temporalWindow)
            
            temporalWindow = eachPossession.moments[temporalWindowStartIndex+1-remainder:temporalWindowStartIndex]
            eachPossession.temporalWindows.insert(0,temporalWindow)

            print("Added Temporal Windows of Possesion with Terminal Action")

        



createTemporalWindows(possessions)

print()