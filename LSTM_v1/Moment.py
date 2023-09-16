# A moment looks like this : [x1,y1,x2,y2, .... x5,y5, bx,by,bz, shot_clock, game_clock]  :  LABEL

# Each moment should have :
    # 1) 25-D array with all the necessary spatio-temporal data
    # 2) A label

# Line from CSV : "[{'x': 81.67902, 'y': 18.37563}, {'x': 89.80558, 'y': 31.86329}, {'x': 80.31479, 'y': 31.49464}, {'x': 69.50902, 'y': 2.11084}, {'x': 79.07408, 'y': 26.42447}, {'x': 74.55269, 'y': 13.83294}, {'x': 83.27226, 'y': 44.45603}, {'x': 79.66844, 'y': 33.28058}, {'x': 70.63343, 'y': 0.09149}, {'x': 59.74416, 'y': 33.44953}]","{'x': 89.12535, 'y': 25.66681, 'z': 8.61488}",10.95,706.98,1


import ast

class Moment:

    def __init__(self, lineFromCSV):    
        self.lineFromCSV = lineFromCSV
        self.momentArray = []
        self.momentLabel = 0  # Initialize the label to 0

        self.playerLocations = ast.literal_eval(self.lineFromCSV[0])
        self.ballLocation = ast.literal_eval(self.lineFromCSV[1])
        self.shotClock = float(self.lineFromCSV[2])
        self.gameClock = float(self.lineFromCSV[3])

    def fillMomentInfoFromCSVLine(self):
        # Extract the string containing CSV data from the tuple


        

        label = self.lineFromCSV[4]
        
        # add player locations

        for eachPlayer in self.playerLocations:
            self.momentArray.append(float(eachPlayer["x"]))
            self.momentArray.append(float(eachPlayer["y"]))

        # add ball location
        
        self.momentArray.append(float(self.ballLocation["x"]))
        self.momentArray.append(float(self.ballLocation["y"]))
        self.momentArray.append(float(self.ballLocation["z"]))

        # add shot and game clock

        self.momentArray.append(float(self.shotClock))
        self.momentArray.append(float(self.gameClock))


        # add the label

        self.momentLabel = float(label)

# Example usage

# if __name__ == "__main__":
#     csv_line = "[{'x': 81.67902, 'y': 18.37563}, {'x': 89.80558, 'y': 31.86329}, {'x': 80.31479, 'y': 31.49464}, {'x': 69.50902, 'y': 2.11084}, {'x': 79.07408, 'y': 26.42447}, {'x': 74.55269, 'y': 13.83294}, {'x': 83.27226, 'y': 44.45603}, {'x': 79.66844, 'y': 33.28058}, {'x': 70.63343, 'y': 0.09149}, {'x': 59.74416, 'y': 33.44953}]","{'x': 89.12535, 'y': 25.66681, 'z': 8.61488}",10.95,706.98,1
#     m1 = Moment(csv_line)
#     m1.fillMomentInfoFromCSVLine()

#     print(m1.momentArray)
#     print(m1.momentLabel)
