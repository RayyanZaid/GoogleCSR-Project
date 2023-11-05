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


@print_error_and_continue
def plotLoss(history,modelName):
    # Plot training and validation loss
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot Over Epochs')
    plt.legend()
    plt.savefig(f'{modelName}/loss_plot.png')
    plt.show()

@print_error_and_continue
def plotAccuracy(history,modelName):
    # Plot training and validation accuracy
    plt.plot(history['categorical_accuracy'], label='Training Accuracy')
    plt.plot(history['val_categorical_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Plot Over Epochs')
    plt.legend()
    plt.savefig(f'Graphs_{modelName}/accuracy_plot.png')
    plt.show()

@print_error_and_continue
def plotLearningCurve(history,X_train,modelName):
    train_loss = []  # To store training loss
    val_loss = []    # To store validation loss
    train_acc = []   # To store training accuracy
    val_acc = []     # To store validation accuracy

    train_loss.extend(history['loss'])
    val_loss.extend(history['val_loss'])
    train_acc.extend(history['categorical_accuracy'])
    val_acc.extend(history['val_categorical_accuracy'])

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
    plt.savefig(f'Graphs_{modelName}/learning_curves.png')
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
def plot_label_frequency(label_counts_actual, label_counts_predicted, modelName):
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

    plt.savefig(f'Graphs_{modelName}/label_frequency.png')
    plt.show()

@print_error_and_continue
def plot_percent_error(label_counts_actual, label_counts_predicted, modelName):
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

    plt.savefig(f'Graphs_{modelName}/percent_error.png')
    plt.show()

@print_error_and_continue
def plotLabelFreqAndPercentErr(history, X_test, y_test, model, modelName):
    y_pred = model.predict(X_test)
    actual_time_series = y_test
    predicted_time_series = y_pred

    actual_time_series = np.array(actual_time_series)
    predicted_time_series = np.array(predicted_time_series)

    # Extract the selected index with the highest probability for each time step
    selected_indices = [np.argmax(probabilities) for probabilities in predicted_time_series]

    # Calculate and plot label frequency and percent error
    label_counts_actual, label_counts_predicted = count_label_frequency(actual_time_series, selected_indices)
    plot_label_frequency(label_counts_actual, label_counts_predicted, modelName)
    plot_percent_error(label_counts_actual, label_counts_predicted, modelName)

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from sklearn.calibration import calibration_curve

def plot_reliability_curve(model, X_test, y_test, class_index, modelName, n_bins=10):
    # Get the predicted probabilities for each class
    y_prob = model.predict(X_test)[:, class_index]

    # Convert multi-class labels to binary labels for the specific class
    pos_label = class_index  # Use the class index as the positive label
    y_true_binary = (y_test == pos_label).astype(int)

    # Create the reliability curve
    prob_true, prob_pred = calibration_curve(y_true_binary, y_prob, n_bins=n_bins)

    # Plot the reliability curve
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', linestyle='--', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='red')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Reliability Curve')
    plt.grid()
    plt.savefig(f'Graphs_{modelName}/reliability_curve{mapping[class_index]}.png')
    plt.show()








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
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32))  # You can add more LSTM layers if needed
    
    # Dense layers
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    
    # Output layer
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


# Initialize empty numpy arrays
X_train = np.empty((0, *expected_shape))
y_train_encoded = np.empty((0, *expected_y_shape))
X_valid = np.empty((0, *expected_shape))
y_valid_encoded = np.empty((0, *expected_y_shape))
X_test = np.empty((0, *expected_shape))
y_test = []

# Iteration through pkl files in the directory


startFile = 0
endFile = 50
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

            # Extend the corresponding arrays with data from pickle files
            X_train = np.concatenate((X_train, training_data_for_pickle['X_train']), axis=0)
            y_train_encoded = np.concatenate((y_train_encoded, training_data_for_pickle['y_train_encoded']), axis=0)
            X_valid = np.concatenate((X_valid, training_data_for_pickle['X_valid']), axis=0)
            y_valid_encoded = np.concatenate((y_valid_encoded, training_data_for_pickle['y_valid_encoded']), axis=0)
            X_test = np.concatenate((X_test, training_data_for_pickle['X_test']), axis=0)
            y_test.append(training_data_for_pickle['y_test'])


        file_count += 1

            
        

# Check the final dimensions
# if X_train.shape != (num_files, *expected_shape):
#     raise ValueError(f"Final X_train shape is unexpected: {X_train.shape}")
# if X_valid.shape != (num_files, *expected_shape):
#     raise ValueError(f"Final X_valid shape is unexpected: {X_valid.shape}")
# if X_test.shape != (num_files, *expected_shape):
#     raise ValueError(f"Final X_test shape is unexpected: {X_test.shape}")






print("Done")

def trainModel(model, directory):
    cp = ModelCheckpoint(directory, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.0001
    )

    model.compile(
        loss=CategoricalCrossentropy(),
        optimizer=Adam(learning_rate=0.0001),
        metrics=[CategoricalAccuracy()]
    )

    history = model.fit(
        X_train,
        y_train_encoded,
        validation_data=(X_valid, y_valid_encoded),
        epochs=10,
        callbacks=[cp, reduce_lr],
        batch_size=32
    )

    

    with open(f'{directory}_history.pkl', 'wb') as file:
        pickle.dump(history, file)
    
    model.save(os.path.join(directory, f'{directory}_savedModel.h5'))

    return history

# Create and train the model
model = create1DConvLSTM()
name = "1D_Conv_LSTM_v5"

# Define the directory path
directory = f'Graphs_{name}'

# Check if the directory exists, and if not, create it
if not os.path.exists(directory):
    os.makedirs(directory)

history = trainModel(model, name)

# model = load_model("stacked_LSTM")



# graphs

plotLoss(history.history,name)
plotAccuracy(history.history,name)
plotLearningCurve(history.history, X_train,name)
plotLabelFreqAndPercentErr(history.history,X_test,y_test,model,name)

# for eachLabel in mapping:
#     plot_reliability_curve(model,X_test,y_test,int(eachLabel), name)