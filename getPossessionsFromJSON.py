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


from Moment import Moment
from typing import List
from Possession import Possession
import pandas as pd


import pandas as pd


import re
# Goal is to store each moment data in a CSV file (annotated)

# ex: playerLocations , ballLocation , shotClock --> annotation
from enum import Enum
from nba_api.stats.endpoints import playbyplay
from globals import MOMENT_SIZE, print_error_and_continue

@print_error_and_continue
def add_seconds_to_time(time_str, seconds_to_add):
    # Split the time string into minutes and seconds
    minutes_str, seconds_str = time_str.split(':')
    
    # Convert minutes and seconds to integers
    minutes = int(minutes_str)
    seconds = int(seconds_str)
    
    # Add seconds_to_add to the seconds component
    seconds += seconds_to_add
    
    # Handle overflow if seconds exceed 59
    if seconds >= 60:
        minutes += 1
        seconds -= 60
    
    # Determine the format for the updated time
    if minutes == 0:
        updated_time_str = f"{seconds:02}"
    else:
        updated_time_str = f"{minutes}:{seconds:02}"
    
    # Add '0:' prefix if necessary
    if len(updated_time_str) == 2:
        updated_time_str = f"0:{updated_time_str}"
    
    return updated_time_str



        

class MomentPreprocessingClass:

    def __init__(self, json_path):
        self.json_path = json_path
        pattern = r'\d+'
        data_frame = pd.read_json(json_path)

        self.events = data_frame['events']
        match = re.search(pattern, json_path)
        self.currentTeamPossessionID : int
        if match:
            number_part = match.group()
       

        df_From_NBA_API = playbyplay.PlayByPlay(number_part).get_data_frames()[0]

        self.NBA_API_MAP = {}

        for eachEvent in df_From_NBA_API.values:
            
            quarter = eachEvent[4]
            timeStamp = eachEvent[6]
            eventNum = eachEvent[2]                
            
            if eventNum == 1 or eventNum == 2:

                eventNum = 1 # FG attempt
                # print("Field Goal Attempt")
                timeStamp = add_seconds_to_time(timeStamp, 3)


            elif eventNum == 5:

                eventNum = 4 # turnover
                # print("Turnover")
            
            elif eventNum == 6:

                # check index 7 (team 1) and index 9 (team 2) to see if it's a regular or shooting foul
                
                if eachEvent[7] is not None:
                    if "S.FOUL" in eachEvent[7]:
                        eventNum = 2
                    elif "P.FOUL" in eachEvent[7]:
                        eventNum = 3
                    else:
                        continue
                elif eachEvent[9] is not None:
                    if "S.FOUL" in eachEvent[9]:
                        eventNum = 2
                    elif "P.FOUL" in eachEvent[9]:
                        eventNum = 3
                    else:
                        continue
                else:  # for double techs lol
                    continue


            else:
                continue

            self.NBA_API_MAP[(quarter,timeStamp)] = eventNum

        self.lastAnnotationNum = "0"
        self.lastGameClockNum = "720.00"

    @print_error_and_continue
    def annotateMomentUsingNBA_API(self, quarterOfMoment, secondsUntilEndOfQuarterOfMoment):
        # Truncate the decimal points in secondsUntilEndOfQuarterOfMoment
        secondsUntilEndOfQuarterOfMoment = int(secondsUntilEndOfQuarterOfMoment)

        # Calculate minutes and seconds
        minutes = secondsUntilEndOfQuarterOfMoment // 60
        seconds = secondsUntilEndOfQuarterOfMoment % 60

        # Format the result as a string in "mm:ss" format
        if minutes >= 10:
            timeStampOfMoment = f"{minutes:02}:{seconds:02}"
        else:
            timeStampOfMoment = f"{minutes}:{seconds:02}"

        if timeStampOfMoment == '12:00':
            return 0

        # check if the time stamp of NBA API event matches the time stamp of the moment
        # for now round down each second
        if (quarterOfMoment, timeStampOfMoment) in self.NBA_API_MAP:
            eventTypeNum = self.NBA_API_MAP[(quarterOfMoment, timeStampOfMoment)]
            return eventTypeNum
        else:
            return 0


    @print_error_and_continue
    def getData(self,json_path):

        

            
        rowNumber = 1

        possessionCounter = 0

        
        allPossessions : List[Possession] = []
        currentPossession: Possession = Possession()
        currentShotClock = 0
        previousShotClock = 25

        currentGameClock = 0
        previousGameClock = 730

        currentQuarter = 1
        prevQuarter = 1

        momentPreprocessingClass = MomentPreprocessingClass(json_path)

        afterTerminalAction = False

        
        
        for eachEvent in self.events:
            homeTeamID : int = eachEvent['home']['teamid']
            visitorTeamID : int = eachEvent['visitor']['teamid']

            moments = eachEvent["moments"]

            for eachMoment in moments:
                # print(possessionCounter)
                quarterFromMoment = eachMoment[0]
                currentQuarter = quarterFromMoment

                if currentQuarter != prevQuarter:
                    previousGameClock = currentGameClock
                    previousShotClock = currentShotClock
                    prevQuarter = currentQuarter
                    allPossessions.append(currentPossession)
                    currentPossession = Possession()
                
                momentObject = Moment(eachMoment)
                momentObject.whichSideIsOffensive()
                momentObject.whichTeamHasPossession()

                if momentObject.possessingTeamID == None:
                    continue
                
                if momentObject.possessingTeamID == homeTeamID:
                    currentPossession.homePossessionCounter += 1
                else:
                    currentPossession.visitorPossessionCounter += 1

                if currentPossession.homePossessionCounter >= currentPossession.visitorPossessionCounter:
                    self.currentTeamPossessionID = homeTeamID
                else:
                    self.currentTeamPossessionID = visitorTeamID

                momentObject.fillMomentFromJSON(self.currentTeamPossessionID)
                
                currentShotClock = momentObject.shot_clock
                currentGameClock = momentObject.game_clock
                
                if previousGameClock == None or previousShotClock == None:
                    previousGameClock = currentGameClock
                    previousShotClock = currentShotClock
                    prevQuarter = currentQuarter
                    continue
   
                if currentGameClock > previousGameClock:
                    previousShotClock = currentShotClock
                    previousGameClock = currentGameClock
                    prevQuarter = currentQuarter
                    continue

                
                if(momentPreprocessingClass.lastGameClockNum == momentObject.game_clock or momentObject.game_clock == None or momentObject.shot_clock == None):
                    previousShotClock = currentShotClock
                    previousGameClock = currentGameClock
                    prevQuarter = currentQuarter
                    continue
                else:
                    momentPreprocessingClass.lastGameClockNum = momentObject.game_clock

                isTerminalAction = False

                label = momentPreprocessingClass.annotateMomentUsingNBA_API(momentObject.quarterNum , momentObject.game_clock)

                momentObject.momentLabel = label

                if momentObject.momentLabel != 0:
                    isTerminalAction = True

                


                if afterTerminalAction:

                    if currentShotClock > previousShotClock:
                        afterTerminalAction = False
                        
                        if len(momentObject.momentArray) == MOMENT_SIZE:
                            currentPossession.addMoment(momentObject)

                        if len(currentPossession.moments) >= 75:
                            allPossessions.append(currentPossession)
                            possessionCounter+=1
        

                        currentPossession = Possession()
                            
                    

                    if len(momentObject.momentArray) == MOMENT_SIZE:
                        currentPossession.addMoment(momentObject)
                        
                    previousShotClock = currentShotClock
                    previousGameClock = currentGameClock
                    prevQuarter = currentQuarter
                        
                    rowNumber += 1
                    continue


                if currentShotClock > previousShotClock:
                    

                    if len(currentPossession.moments) >= 75:
                        allPossessions.append(currentPossession)
                        possessionCounter+=1
                    

                if isTerminalAction:
                        afterTerminalAction = True
                        rowNumber += 1
                        if len(momentObject.momentArray) == MOMENT_SIZE:
                            currentPossession.addMoment(momentObject)
                        currentPossession.terminalActionIndex = len(currentPossession.moments) - 1
                        previousShotClock = currentShotClock
                        previousGameClock = currentGameClock
                        prevQuarter = currentQuarter
                        continue

                if currentShotClock == previousShotClock:
                    previousShotClock = currentShotClock
                    previousGameClock = currentGameClock
                    prevQuarter = currentQuarter
                    rowNumber += 1
                    continue

                
                if len(momentObject.momentArray) == MOMENT_SIZE:
                    currentPossession.addMoment(momentObject)


                rowNumber += 1
                previousShotClock = currentShotClock
                previousGameClock = currentGameClock
                prevQuarter = currentQuarter

        return allPossessions





