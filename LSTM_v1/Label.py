

# 1 --> attempted field goal
# 2 --> shooting foul
# 3 --> non-shooting foul
# 4 --> turnover
# 5 --> null


labelMap = {
    1 : "attempted field goal",
    2 : "shooting foul",
    3 : "non-shooting foul",
    4 : "turnover",
    5 : "null"
}


# Purpose of this class : to take the labels from the CSV file and encode them for the LSTM

class Label:

    def __init__(self, labelFromCSV):

        self.label : int = 0
    
    def encodeFromCSV(self):
        print()