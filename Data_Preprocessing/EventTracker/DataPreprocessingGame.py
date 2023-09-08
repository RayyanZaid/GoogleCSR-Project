import pandas as pd
from Event import Event
import csv

# Goal is to store each moment data in a CSV file (annotated)

# ex: playerLocations , ballLocation , shotClock --> annotation

class DataPreprocessing:

    def __init__(self, json_path):
        self.json_path = json_path
        self.events = None

    def read_json(self):
        data_frame = pd.read_json(self.json_path)
        self.events = data_frame['events']

    def iterateThroughEvents(self):
        moment_data_list = []  # List to store data for each moment

        for eachEvent in self.events:
            eventObject = Event(eachEvent)
            for momentObject in eventObject.moments:
                moment_data = {
                    "annotation": "shot",  # Annotation for each moment (can be modified later)
                    "quarter": momentObject.quarter,
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

                moment_data_list.append(moment_data)

        # Write the moment data to a CSV file
        with open("game.csv", mode="w", newline="") as csv_file:
            fieldnames = ["annotation", "quarter", "shot_clock", "playerLocations", "ballLocation"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for moment_data in moment_data_list:
                writer.writerow(moment_data)

# Example usage:
obj = DataPreprocessing(r"C:\Users\rayya\OneDrive\Desktop\GoogleCSR-Project\Data_Preprocessing\EventTracker\0021500524.json")
obj.read_json()
obj.iterateThroughEvents()
