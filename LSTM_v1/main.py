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


# %% Cell 0
from typing import List
import numpy as np
from Possession import Possession
from getPossessionsFromCSV import getData
from Moment import Moment
import helperFunctions 
from MomentPreprocessing.MomentPreprocessingMain import MomentPreprocessingClass

# %% Cell 1

# 1) Read JSON and preprocess the data into a CSV

# Already preprocessed into CSV

obj = MomentPreprocessingClass(r"C:\Users\rayya\OneDrive\Desktop\GoogleCSR-Project\Datasets\0021500524.json")
obj.read_json()
obj.iterateThroughEvents()

# 2) Read from CSV and create possessions

# %% Cell 2
possessions : List[Possession] = getData()


# %% Cell 3
# 3) Go through possessions and create Temporal Windows


# Now call the function
helperFunctions.createTemporalWindows(possessions)

# %% Cell 4

# 4) Process data so that it's in a input matrix : output vector format


# This function:
    # Input : List of Possessions with Temporal Windows updated
    # Ouput : the input matrix and output vector

inputMatrix, outputVector = helperFunctions.processDataForLSTM(possessions)


# %% Cell 5
import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame from the outputVector with a "Label" column
df_output = pd.DataFrame(outputVector, columns=["Label"])

# Calculate the percentages
percentage_counts = df_output["Label"].value_counts(normalize=True)

mapping = {
    0.0: "null",
    1.0: "FG Attempt",
    2.0: "Shooting Foul",
    3.0: "Nonshooting Foul",
    4.0: "Turnover"
}

# Map the labels using the mapping dictionary
mapped_labels = percentage_counts.index.map(mapping)

# Create a pie chart
plt.pie(percentage_counts, labels=mapped_labels, autopct='%1.1f%%', startangle=140)

# Add a title
plt.title("Pie Chart of Label Percentages")

# Show the plot
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# %% Cell 6 -- Create numpy arrays



inputMatrix = np.array(inputMatrix)   # SHAPE: number of windows 1500, 128, 25
outputVector = np.array(outputVector) # SHAPE: number of windows 1500, 1 

# print(inputMatrix.shape)
# print(outputVector.shape)


# %% Cell 7 -- Split into Train, Test, and Validation

from sklearn.model_selection import train_test_split

numberOfTemporalWindows = inputMatrix.shape[0]

# 80% Train
# 10% Validation
# 10% Test

X_train, X_rem, y_train, y_rem = train_test_split(inputMatrix,outputVector,train_size=0.8,random_state=42)

X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

# %% 


# %% Cell 8 -- Import Keras Libraries

from keras.models import Sequential # Sequential Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint # To save model
from keras.losses import MeanSquaredError # For loss function
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam

# %% Cell 9 -- Create Model

model1 = Sequential()
model1.add(InputLayer((128,25)))
model1.add((LSTM(64)))
model1.add(Dense(8,'relu'))
model1.add(Dense(5,activation='softmax'))

model1.summary()

# %% Cell 10 -- Model Callbacks and Compilation

cp = ModelCheckpoint('model1/' , save_best_only=True) # saves model with lowest validation loss
model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics = [RootMeanSquaredError()]) # higher the learning rate, the faster the model will try to decrease the loss function

# %% Cell 11 -- Fitting

model1.fit(X_train, y_train, validation_data=(X_valid, y_valid) , epochs = 10, callbacks=[cp])

# %% Cell 12 -- Using the model

from keras.models import load_model

model1 = load_model('model1/')


train_predictions = model1.predict(X_train).flatten()
# train_results = pd.DataFrame(data={'Train Predictions' : train_predictions , 
#                                    'Actuals' : y_train
#                                 })
# %%
