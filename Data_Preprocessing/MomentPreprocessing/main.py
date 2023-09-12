import pandas as pd
from Event import Event
import csv
import re
# Goal is to store each moment data in a CSV file (annotated)

# ex: playerLocations , ballLocation , shotClock --> annotation
from enum import Enum
from nba_api.stats.endpoints import playbyplay

class EventMsgType(Enum):
    FIELD_GOAL_MADE = 1
    FIELD_GOAL_MISSED = 2
    FREE_THROWfree_throw_attempt = 3
    REBOUND = 4
    TURNOVER = 5
    FOUL = 6
    VIOLATION = 7
    SUBSTITUTION = 8
    TIMEOUT = 9
    JUMP_BALL = 10
    EJECTION = 11
    PERIOD_BEGIN = 12
    PERIOD_END = 13

class MomentPreprocessing:

    def __init__(self, json_path):
        self.json_path = json_path
        self.events = None
        pattern = r'\d+'

        match = re.search(pattern, json_path)

        if match:

            number_part = match.group()
            print("Extracted number part:", number_part)
        else:
            print("No number found in the text.")

        self.df_From_NBA_API = df = playbyplay.PlayByPlay(number_part).get_data_frames()[0]

        self.dfEventPointer = 0


    def read_json(self):
        data_frame = pd.read_json(self.json_path)
        self.events = data_frame['events']

    def iterateThroughEvents(self):
        moment_data_list = []  # List to store data for each moment

        for eachEvent in self.events:
            eventObject = Event(eachEvent)

            for momentObject in eventObject.moments:
                moment_data = {
                    "annotation": "0",  # Annotation for each moment (can be modified later)
                    # "quarter": momentObject.quarter,

                    "game_clock" : momentObject.game_clock,
                    "shot_clock": momentObject.shot_clock,
                    
                    "playerLocations": [],
                    "ballLocation": []
                }



                # Extract player locations (x, y) for the first 20 elements
                for i in range(len(momentObject.players)):
                    player = momentObject.players[i]
                    moment_data["playerLocations"].append({"x": player.x, "y": player.y})

                # Extract ball coordinates (x, y, z) for the next 3 elements
                ball = momentObject.ball
                moment_data["ballLocation"] = {"x": ball.x, "y": ball.y, "z": ball.radius}

                


                # also add the label

                quarter = momentObject.quarter
                secondsUntilEndOfQuarter = momentObject.game_clock
                
                if self.dfEventPointer < len(self.df_From_NBA_API.values) and secondsUntilEndOfQuarter > 1:
                    annotation = self.annotateMomentUsingNBA_API(quarter,secondsUntilEndOfQuarter)

                if annotation == None:
                    moment_data["annotation"] = "null"
                else:
                    moment_data["annotation"] = annotation
                moment_data_list.append(moment_data)

        # Write the moment data to a CSV file
        with open("Data_Preprocessing/moments.csv", mode="w", newline="") as csv_file:
            fieldnames = ["playerLocations", "ballLocation", "shot_clock" , "game_clock" , "annotation"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for moment_data in moment_data_list:
                writer.writerow(moment_data)
                


    def annotateMomentUsingNBA_API(self,quarterOfMoment, secondsUntilEndOfQuarterOfMoment):
        

        # Truncate the decimal points in secondsUntilEndOfQuarterOfMoment
        secondsUntilEndOfQuarterOfMoment = int(secondsUntilEndOfQuarterOfMoment)

        # Calculate minutes and seconds
        minutes = secondsUntilEndOfQuarterOfMoment // 60
        seconds = secondsUntilEndOfQuarterOfMoment % 60

        # Format the result as a string in "mm:ss" format
        timeStampOfMoment = f"{minutes:02}:{seconds:02}"


        if timeStampOfMoment == '12:00':
            return None

        current_NBA_API_Event = self.df_From_NBA_API.values[self.dfEventPointer]
        
        # check if the time stamp of NBA API event matches the time stamp of the moment
        # for now round down each second

        timeStampOfCurrent_NBA_API_Event = current_NBA_API_Event[6]

        while timeStampOfCurrent_NBA_API_Event == "12:00":
            self.dfEventPointer += 1
            current_NBA_API_Event = self.df_From_NBA_API.values[self.dfEventPointer]
            timeStampOfCurrent_NBA_API_Event = current_NBA_API_Event[6]


        if timeStampOfMoment == timeStampOfCurrent_NBA_API_Event:
            eventTypeNum = current_NBA_API_Event[2]
            self.dfEventPointer += 1
            return eventTypeNum
        else:
            return None


        

# Example usage:
obj = MomentPreprocessing(r"C:\Users\rayya\OneDrive\Desktop\GoogleCSR-Project\Data_Preprocessing\MomentPreprocessing\0021500494.json")
obj.read_json()
obj.iterateThroughEvents()
