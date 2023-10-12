import pickle
import os
from globals import print_error_and_continue, WINDOW_SIZE, MOMENT_SIZE
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential # Sequential Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau  # save model and learning rate annealing
from keras.losses import CategoricalCrossentropy # For loss function
from keras.metrics import CategoricalAccuracy
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.losses import CategoricalCrossentropy
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
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot Over Epochs')
    plt.legend()
    plt.savefig('Graphs/loss_plot.png')
    plt.show()

@print_error_and_continue
def plotAccuracy(history):
    # Plot training and validation accuracy
    plt.plot(history['categorical_accuracy'], label='Training Accuracy')
    plt.plot(history['val_categorical_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Plot Over Epochs')
    plt.legend()
    plt.savefig('Graphs/accuracy_plot.png')
    plt.show()

@print_error_and_continue
def plotLearningCurve(history,X_train):
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
def plotTimeSeriesWithFrequencyAndPercentError(history, X_test, y_test, model):
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


pkl_directory = r'C:\Users\rayya\Desktop\GoogleCSR-Project\training_history_groups'


data = {     'X_train' : [],
            'X_test'  : [],
            'y_train_encoded' : [],
            'y_test'  : [],
            'X_valid' : [],
            'y_valid_encoded' : []}

# Iteration through pkl files in the directory


for filename in os.listdir(pkl_directory):
    if filename.endswith('.pkl'):
        file_path = os.path.join(pkl_directory, filename)
        with open(file_path, 'rb') as file:
            training_data_for_pickle = pickle.load(file)
            # Merge the data from each file into the combined_history dictionary
            for key in data:
                data[key].extend(training_data_for_pickle[key])

X_train = np.array(data['X_train'])
y_train_encoded = np.array(data['y_train_encoded'])
X_valid = np.array(data['X_valid'])
y_valid_encoded = np.array(data['y_valid_encoded'])
X_test = np.array(data['X_test'])
y_test = np.array(data['y_test'])


print("Done")



# from keras.models import load_model
from keras.models import Sequential # Sequential Model

# # FIX LATER 
# # Create model
from keras.models import load_model

model_directory = r"model2"
model2 : Sequential

if not os.path.exists(model_directory):
    model2 = createModel()
else:
    model2 = load_model(model_directory)

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


plotLoss(history.history)
plotAccuracy(history.history)
plotLearningCurve(history.history,X_train)
plotTimeSeriesWithFrequencyAndPercentError(history.history,X_test,y_test,model2)