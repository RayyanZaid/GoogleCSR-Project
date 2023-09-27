# A moment looks like this : [x1,y1,x2,y2, .... x5,y5, bx,by,bz, shot_clock, game_clock]  :  LABEL

# Each moment should have :
    # 1) 25-D array with all the necessary spatio-temporal data
    # 2) A label

# Line from CSV : "[{'x': 81.67902, 'y': 18.37563}, {'x': 89.80558, 'y': 31.86329}, {'x': 80.31479, 'y': 31.49464}, {'x': 69.50902, 'y': 2.11084}, {'x': 79.07408, 'y': 26.42447}, {'x': 74.55269, 'y': 13.83294}, {'x': 83.27226, 'y': 44.45603}, {'x': 79.66844, 'y': 33.28058}, {'x': 70.63343, 'y': 0.09149}, {'x': 59.74416, 'y': 33.44953}]","{'x': 89.12535, 'y': 25.66681, 'z': 8.61488}",10.95,706.98,1




        
class Moment:

    def __init__(self, jsonMomentArray):    
        self.jsonMomentArray = jsonMomentArray
        self.momentArray = []
        self.momentLabel = 0  # Initialize the label to 0

        self.players = jsonMomentArray[5][1:11]
        self.ball = jsonMomentArray[5][0][2:5]
        self.shot_clock = jsonMomentArray[3]
        self.game_clock = jsonMomentArray[2]

        self.quarterNum = jsonMomentArray[0]
        print()

        


    def fillMomentFromJSON(self):

        
        # storing all 25 values into the moment Array (no label yet)

        for eachPlayer in self.players:
            self.momentArray.append(float(eachPlayer[2]))
            self.momentArray.append(float(eachPlayer[3]))

        # add ball location
        
        self.momentArray.append(float(self.ball[0]))
        self.momentArray.append(float(self.ball[1]))
        self.momentArray.append(float(self.ball[2]))

        # add shot and game clock

        if self.shot_clock == None: 
            self.momentArray.append(float(24.0))
        else:
            self.momentArray.append(float(self.shot_clock))


        if self.game_clock == None: 
            self.momentArray.append(float(720.00))
        else:
            self.momentArray.append(float(self.game_clock))


        # add the label

    def whichSideIsOffensive(self) -> str:

        # halfcourt is at x = 47.0

        # count how many are less than 47.0
        
        counter = 0

        for eachPlayer in self.players:
            x = eachPlayer[2]

            if x <= 47.0:
                counter +=1

        if counter >= 5:
            return "Left"
        else:
            return "Right"
    


# Example usage

# if __name__ == "__main__":
#     csv_line = "[{'x': 81.67902, 'y': 18.37563}, {'x': 89.80558, 'y': 31.86329}, {'x': 80.31479, 'y': 31.49464}, {'x': 69.50902, 'y': 2.11084}, {'x': 79.07408, 'y': 26.42447}, {'x': 74.55269, 'y': 13.83294}, {'x': 83.27226, 'y': 44.45603}, {'x': 79.66844, 'y': 33.28058}, {'x': 70.63343, 'y': 0.09149}, {'x': 59.74416, 'y': 33.44953}]","{'x': 89.12535, 'y': 25.66681, 'z': 8.61488}",10.95,706.98,1
#     m1 = Moment(csv_line)
#     m1.fillMomentInfoFromCSVLine()

#     print(m1.momentArray)
#     print(m1.momentLabel)
