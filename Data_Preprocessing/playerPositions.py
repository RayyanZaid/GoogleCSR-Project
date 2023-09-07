import json
import math

# Define the path to your JSON file
json_file_path = r"C:\Users\rayya\OneDrive\Desktop\GoogleCSR-Project\Data_Preprocessing\0021500524.json"

# Open and read the JSON file
with open(json_file_path, "r") as json_file:
    data = json.load(json_file)

# Initialize variables to keep track of the current second and quarter
current_second = None
current_quarter = None

# Iterate through each event
for event in data["events"]:
    # Iterate through each moment in the event

    
    for moment in event["moments"]:
        # Extract the timestamp, seconds left in the current quarter, and quarter number
        timestamp = moment[1]
        seconds_left = moment[2]
        quarter = moment[0]

        # Round down the current second to the nearest whole second
        current_second_rounded = math.floor(seconds_left)

        # Check if the timestamp represents a new second
        if current_second_rounded != current_second:
            current_second = current_second_rounded
            current_quarter = quarter
            # Extract the list of player positions from the moment
            player_positions = moment[5]

            # Iterate through player positions to find playerId 201939
            for player_data in player_positions:
                team_id, player_id, x, y, z = player_data
                if player_id == 201939:
                    # Print x and y coordinates, along with seconds left in the quarter
                    print(f"Quarter: {current_quarter}, Timestamp: {timestamp}, X: {x}, Y: {y}, Seconds Left: {current_second}")
