# 1) Read JSON and preprocess the data into a CSV

# 2) Read from CSV and create possessions

# 3) Go through possessions and create Temporal Windows

# 4) Input each temporal window into the LSTM


from MomentPreprocessing.MomentPreprocessingMain import MomentPreprocessingClass

# 1) Read JSON and preprocess the data into a CSV


obj = MomentPreprocessingClass(r"C:\Users\rayya\OneDrive\Desktop\GoogleCSR-Project\Datasets\0021500524.json")
obj.read_json()
obj.iterateThroughEvents()