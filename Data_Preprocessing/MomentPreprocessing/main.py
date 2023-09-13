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
    if minutes >= 10:
        updated_time_str = f"{minutes}:{seconds:02}"
    else:
        updated_time_str = f"{minutes}:{seconds}"
    
    return updated_time_str







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

        df_From_NBA_API = playbyplay.PlayByPlay(number_part).get_data_frames()[0]

        self.NBA_API_MAP = {}

        for eachEvent in df_From_NBA_API.values:
            
            quarter = eachEvent[4]
            timeStamp = eachEvent[6]
            eventNum = eachEvent[2]

        # Field Goal Attempt
            if eventNum == 1 or eventNum == 2:
                print("Field Goal Attempt")
                timeStamp = add_seconds_to_time(timeStamp, 1)

            elif eventNum == 5:
                print("Turnover")
            
            elif eventNum == 6:
                print("Foul")

            else:
                continue

            self.NBA_API_MAP[(quarter,timeStamp)] = eventNum


            self.lastAnnotationNum = "0"
            self.lastGameClockNum = "720.00"


    def read_json(self):
        data_frame = pd.read_json(self.json_path)
        self.events = data_frame['events']

    def iterateThroughEvents(self):
        # Open the CSV file in append mode
        with open("Data_Preprocessing/moments.csv", mode="a", newline="") as csv_file:


            fieldnames = ["playerLocations", "ballLocation", "shot_clock", "game_clock", "annotation"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            # Iterate through events and moments
            for eachEvent in self.events:
                eventObject = Event(eachEvent)

                for momentObject in eventObject.moments:
                    moment_data = {
                        "annotation": "null",
                        "game_clock": momentObject.game_clock,
                        "shot_clock": momentObject.shot_clock,
                        "playerLocations": [],
                        "ballLocation": []
                    }

                    if(self.lastGameClockNum == momentObject.game_clock or momentObject.game_clock == None or momentObject.shot_clock == None):
                        continue
                    else:
                        self.lastGameClockNum = momentObject.game_clock


                    for i in range(len(momentObject.players)):
                        player = momentObject.players[i]
                        moment_data["playerLocations"].append({"x": player.x, "y": player.y})

                    ball = momentObject.ball
                    moment_data["ballLocation"] = {"x": ball.x, "y": ball.y, "z": ball.radius}

                    quarter = momentObject.quarter
                    secondsUntilEndOfQuarter = momentObject.game_clock

                    annotation = self.annotateMomentUsingNBA_API(quarter, secondsUntilEndOfQuarter)
                    
                    if(self.lastAnnotationNum != annotation):
                        moment_data["annotation"] = annotation
                    
                    else:
                       moment_data["annotation"] = "null" 

                    self.lastAnnotationNum = annotation

                    

                    print(f"annotation: {annotation}, moment_data: {moment_data}")

                    # Write the moment data to the CSV file immediately
                    writer.writerow(moment_data)
                


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
            return "null"

        # check if the time stamp of NBA API event matches the time stamp of the moment
        # for now round down each second
        if (quarterOfMoment, timeStampOfMoment) in self.NBA_API_MAP:
            eventTypeNum = self.NBA_API_MAP[(quarterOfMoment, timeStampOfMoment)]
            return eventTypeNum
        else:
            return "null"



        

# Example usage:
obj = MomentPreprocessing(r"C:\Users\rayya\OneDrive\Desktop\GoogleCSR-Project\Datasets\0021500524.json")
obj.read_json()
obj.iterateThroughEvents()
