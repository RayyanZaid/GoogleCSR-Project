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


# 5) Input each temporal window into the LSTM and Train

from typing import List
import numpy as np
from Possession import Possession
from getPossessionsFromCSV import getData
from Moment import Moment
import helperFunctions 



# 1) Read JSON and preprocess the data into a CSV

# Already preprocessed into CSV

# obj = MomentPreprocessingClass(r"C:\Users\rayya\OneDrive\Desktop\GoogleCSR-Project\Datasets\0021500524.json")
# obj.read_json()
# obj.iterateThroughEvents()

# 2) Read from CSV and create possessions

possessions : List[Possession] = getData()


# 3) Go through possessions and create Temporal Windows


# Now call the function
helperFunctions.createTemporalWindows(possessions)


# 4) Process data so that it's in a input matrix : output vector format


# This function:
    # Input : List of Possessions with Temporal Windows updated
    # Ouput : the input matrix and output vector

inputMatrix, outputVector = helperFunctions.processDataForLSTM(possessions)

# inputMatrix = np.array(inputMatrix)   # SHAPE: number of windows, 128, 25
# outputVector = np.array(outputVector) # SHAPE: 5,1 

# print(inputMatrix.shape)
# print(outputVector.shape)



# 5) Input each temporal window into the LSTM

helperFunctions.loadAndTrainLSTM(inputMatrix,outputVector)
helperFunctions.predict(inputMatrix)