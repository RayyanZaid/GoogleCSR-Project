import pickle
import os
from globals import print_error_and_continue, WINDOW_SIZE, MOMENT_SIZE
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential # Sequential Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping  # save model and learning rate annealing
from keras.losses import CategoricalCrossentropy # For loss function
from keras.metrics import CategoricalAccuracy
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.optimizers import Adam






from keras.models import Sequential
from keras.layers import InputLayer, LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

# Define your data, X_train, y_train_encoded, X_valid, y_valid_encoded, X_test, y_test here
# pkl_directory = r'C:\Users\rayya\Desktop\GoogleCSR-Project\training_history_groups'
pkl_directory = r'C:\Users\rayya\Desktop\GoogleCSR-Project\training_history_groups_v2_Tis100'


import os
import pickle
import numpy as np

# Convert the lists in the data dictionary to NumPy arrays
# data['X_train'] = np.array(data['X_train'])
# data['y_train_encoded'] = np.array(data['y_train_encoded'])
# data['X_valid'] = np.array(data['X_valid'])
# data['y_valid_encoded'] = np.array(data['y_valid_encoded'])
# data['X_test'] = np.array(data['X_test'])





# X_train = np.array(data['X_train'])
# y_train_encoded = np.array(data['y_train_encoded'])
# X_valid = np.array(data['X_valid'])
# y_valid_encoded = np.array(data['y_valid_encoded'])
# X_test = np.array(data['X_test'])
# y_test = np.array(data['y_test'])



@print_error_and_continue
def createStackedLSTM() -> Sequential:
    model = Sequential()
    model.add(InputLayer((WINDOW_SIZE, MOMENT_SIZE)))
    
    # Stacked LSTM layers
    model.add(LSTM(64, return_sequences=True, activation='relu'))  # return_sequences=True for stacked LSTM
    model.add(LSTM(64, return_sequences=True, activation='relu'))
    model.add(LSTM(64,activation='tanh'))  # You can add more LSTM layers if needed
    
    # Dense layers
    model.add(Dense(75, activation='relu'))
    model.add(Dense(32, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='relu'))
    
    # Output layer
    model.add(Dense(5, activation='softmax'))
    
    model.summary()

    return model


from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from keras.optimizers import Adam

from keras.layers import BatchNormalization
from keras.regularizers import l2

from keras.models import Sequential
from keras.layers import InputLayer, LSTM, Dense, Dropout, GRU

def createGRU_LSTM() -> Sequential:
    model = Sequential()
    model.add(InputLayer((WINDOW_SIZE, MOMENT_SIZE)))
    
    # GRU layer
    model.add(GRU(64, return_sequences=True, activation='relu'))
    
    # LSTM layers
    model.add(LSTM(32, return_sequences=True, activation='relu'))  # return_sequences=True for stacked LSTM
    model.add(LSTM(32, return_sequences=True, activation='relu'))
    model.add(LSTM(32, activation='tanh'))  # You can add more LSTM layers if needed
    
    # Dense layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    # Output layer
    model.add(Dense(5, activation='softmax'))
    
    model.summary()

    return model

def create1DConvLSTM():
    model = Sequential()
    
    # 1D Convolutional Layer
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(WINDOW_SIZE, MOMENT_SIZE)))
    model.add(MaxPooling1D(pool_size=2))
    
    # LSTM layers
    model.add(LSTM(32, return_sequences=True))  # return_sequences=True for stacked LSTM
    model.add(LSTM(32))  # You can add more LSTM layers if needed
    
    # Dense layers
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='relu'))

    model.add(Dense(5, activation='softmax'))
    
    model.summary()
    
    return model



import os
import pickle
import numpy as np
import os
import pickle
import numpy as np

# Define the expected dimensions
expected_shape = (100, 24)
expected_y_shape = (5,)



# Iteration through pkl files in the directory

import os
import pickle
import numpy as np

# Initialize your arrays with an appropriate data type (e.g., float32)
X_train = np.empty((0, *expected_shape), dtype=np.float32)  # Replace 'your_shape' with the actual shape
y_train_encoded = np.empty((0, *expected_y_shape), dtype=np.float32)  # Replace 'your_shape' with the actual shape
X_valid = np.empty((0, *expected_shape), dtype=np.float32)  # Replace 'your_shape' with the actual shape
y_valid_encoded = np.empty((0, *expected_y_shape), dtype=np.float32)  # Replace 'your_shape' with the actual shape
X_test = np.empty((0, *expected_shape), dtype=np.float32)  # Replace 'your_shape' with the actual shape
y_test = []

startFile = 1  # Define your desired range
endFile = 618  # Define your desired range
file_count = 0

for filename in os.listdir(pkl_directory):
    if filename.endswith('.pkl'):
        # Check if the current file is within the desired range
        if file_count < startFile:
            file_count += 1
            continue
        elif file_count >= endFile:
            break

        file_path = os.path.join(pkl_directory, filename)
        with open(file_path, 'rb') as file:
            training_data_for_pickle = pickle.load(file)

            # Convert loaded data to float32 before concatenating
            X_train_data = training_data_for_pickle['X_train'].astype(np.float32)
            y_train_encoded_data = training_data_for_pickle['y_train_encoded'].astype(np.float32)
            X_valid_data = training_data_for_pickle['X_valid'].astype(np.float32)
            y_valid_encoded_data = training_data_for_pickle['y_valid_encoded'].astype(np.float32)
            X_test_data = training_data_for_pickle['X_test'].astype(np.float32)
            y_test_data = [item for item in training_data_for_pickle['y_test']]

            # Extend the corresponding arrays with data from pickle files
            X_train = np.concatenate((X_train, X_train_data), axis=0)
            y_train_encoded = np.concatenate((y_train_encoded, y_train_encoded_data), axis=0)
            X_valid = np.concatenate((X_valid, X_valid_data), axis=0)
            y_valid_encoded = np.concatenate((y_valid_encoded, y_valid_encoded_data), axis=0)
            X_test = np.concatenate((X_test, X_test_data), axis=0)
            y_test.extend(y_test_data)

        file_count += 1

print(f"Done loading games {startFile} to {endFile}")

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.metrics import Mean, CategoricalAccuracy

def trainModel(model, name):
    cp = ModelCheckpoint(name, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.0001
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',  # You can monitor a different validation metric if needed
        patience=5,  # Adjust the patience based on your training needs
        restore_best_weights=True
    )

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.00008),
        metrics=[
            'categorical_accuracy',  # Training accuracy
        
        ]
    )

    history = model.fit(
        X_train,
        y_train_encoded,
        validation_data=(X_valid, y_valid_encoded),
        epochs=50,
        callbacks=[cp, reduce_lr, early_stopping],  # Add early_stopping callback
        batch_size=32
    )

    historiesDirectory = f'{name}_histories'
    file_name = "1st.pkl"

    

    if not os.path.exists(historiesDirectory):
        os.makedirs(historiesDirectory)  # Create the directory if it doesn't exist

    file_path = os.path.join(historiesDirectory, file_name)  # Create the full path including directory

    # Assuming 'history' is the object you want to save

    with open(file_path, 'wb') as file:
        pickle.dump(history, file)
        




# Create and train the model


name = "1D_Conv_LSTM_v8"
if not os.path.exists(name):
    model = create1DConvLSTM()
else:
    model = load_model(name)



trainModel(model, name)

# model = load_model("stacked_LSTM")



# graphs

# plotLoss(history.history,name)
# plotAccuracy(history.history,name)
# plotLearningCurve(history.history, X_train,name)
# plotLabelFreqAndPercentErr(history.history,X_test,y_test,model,name)

# # for eachLabel in mapping:
# #     plot_reliability_curve(model,X_test,y_test,int(eachLabel), name)