import pickle
import os
from globals import print_error_and_continue
import matplotlib.pyplot as plt

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


pkl_directory = r'C:\Users\rayya\Desktop\GoogleCSR-Project\training_history_batches'


combined_history = {'loss': [], 'val_loss': [], 'categorical_accuracy': [], 'val_categorical_accuracy': []}

# Iteration through pkl files in the directory

for filename in os.listdir(pkl_directory):
    if filename.endswith('.pkl'):
        file_path = os.path.join(pkl_directory, filename)
        with open(file_path, 'rb') as file:
            history = pickle.load(file)
            # Merge the data from each file into the combined_history dictionary
            for key in combined_history:
                combined_history[key].extend(history[key])


plotAccuracy(combined_history)
plotLoss(combined_history)

print("Done")
