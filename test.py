import tensorflow as tf

# Get a list of available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    print("Number of GPUs Available: ", len(gpus))
    for gpu in gpus:
        print("Name:", gpu.name)
else:
    print("No GPUs are available on this system.")
