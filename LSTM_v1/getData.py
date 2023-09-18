# Get each Possession
    # 1. Each Possession ends on a terminal action OR
    # 2. Each Possesion ends when the shot clock increases


# Split each Possession into sequences of temporal windows

    # 1. Ensure that ALL the terminal actions are included in the temporal window
        # a) That means starting from a terminal action and going backwards
    # 2. Decide whether temporal windows should overlap or not


# Each temporal window has a sequence of moments


# Each moment should look like this :

# INPUT MATRIX : [ [  [m1], [m2], [m3], ... [m128]     ] ]  where m1 = [x1,y1,x2,y2, .... x5,y5, bx,by,bz, shot_clock, game_clock]          

# LABEL VECTOR : [  [L1], [L2],         ... [L128]       ]


# Access the moments.csv file and extract the temporal windows 

# Temporal Window --> LSTM for training



# the last T=128 moments    +     r (buffer moments)

# t : the time of interest
# T = 128 moments (5 seconds) 
# r = 48 moments (1.5 seconds)

# temporal window = t - (T + r) --> t - r
# or : (t - r) - T --> t - r


# INPUT MATRIX : [ [  [m1], [m2], [m3], ... [m128]     ] ]                   

# LABEL VECTOR : [  [L1], [L2],         ... [L128]       ]

# if a possession ends in null, then scrap the whole possession except for the terminal action (the temporal window at the end of possesion)



# iterate through the CSV file and create possession objects

# keep going through the 

import csv
from Moment import Moment
from typing import List
from Possession import Possession

# Define your functions and classes here

def getData():
    csv_file_path = r"C:\Users\rayya\OneDrive\Desktop\GoogleCSR-Project\Data_Preprocessing\moments.csv"

    with open(csv_file_path, mode='r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        rowNumber = 1

        possessionCounter = 0

        allPossessions : List[Possession] = []
        currentPossession: Possession = Possession()
        currentShotClock = 0
        previousShotClock = 25

        afterTerminalAction = False
        for row in csv_reader:

            moment = Moment(row)
            moment.fillMomentInfoFromCSVLine()
            currentShotClock = moment.shotClock 

            isTerminalAction = False

            if possessionCounter == 2:
                print()

            if moment.momentLabel != 0:
                isTerminalAction = True

            
            if afterTerminalAction:

                if currentShotClock > previousShotClock:
                    afterTerminalAction = False
                    currentPossession.addMoment(moment)
                    possessionCounter+=1
                
                previousShotClock = currentShotClock
                    
                rowNumber += 1
                continue


            if currentShotClock > previousShotClock or isTerminalAction:
                
                allPossessions.append(currentPossession)
                currentPossession = Possession()

                if not isTerminalAction:
                    
                    possessionCounter+=1

                if isTerminalAction:
                    afterTerminalAction = True
                    rowNumber += 1
                    continue

            if currentShotClock == previousShotClock:
                previousShotClock = currentShotClock
                rowNumber += 1
                continue


            currentPossession.addMoment(moment)


            rowNumber += 1
            previousShotClock = currentShotClock

    print("DONE")

if __name__ == "__main__":
    getData()




