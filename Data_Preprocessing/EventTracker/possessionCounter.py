import json
import math

# Define the path to your JSON file
json_file_path = r"C:\Users\rayya\OneDrive\Desktop\GoogleCSR-Project\Datasets\0021500524.json"

# Open and read the JSON file
with open(json_file_path, "r") as json_file:
    data = json.load(json_file)

# Initialize variables to keep track of the current second, quarter, and possessing team
current_second = None
current_quarter = None
current_possessing_team = None
prev_possessing_team = None
current_shot_clock = None
current_time_since_1970 = None

num_possessions = 0
possessing_iterations = 0  # Counter for consecutive iterations with the same possessing team

# Iterate through each event
for event in data["events"]:
    # Initialize a counter to track the number of iterations within the event
    iteration_counter = 0

    # Iterate through each moment in the event
    for moment in event["moments"]:
        # Extract the seconds left in the current quarter, quarter number, and shot clock time
        current_time_since_1970 = moment[1]
        seconds_left = moment[2]
        quarter = moment[0]
        current_shot_clock = moment[3]

        if current_shot_clock is None:
            continue
      
        # Extract the list of player positions from the moment
        player_positions = moment[5]

        # Find the player closest to the ball in the first 5 iterations
        if iteration_counter < 5:
            closest_player = None
            min_distance = float("inf")  # Initialize with a large value

            # Iterate through player positions to calculate distances
            for player_data in player_positions:
                team_id, player_id, x, y, z = player_data
                if player_id != -1:  # Exclude the ball
                    # Calculate the Euclidean distance between the player and the ball
                    distance = math.sqrt((x - player_positions[0][2])**2 + (y - player_positions[0][3])**2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_player = player_id
                        
                        current_possessing_team = team_id

            iteration_counter += 1

        # Check if the possession has changed
        if current_possessing_team != prev_possessing_team:
            current_quarter = quarter
            prev_possessing_team = current_possessing_team
            num_possessions += 1
            possessing_iterations = 1  # Reset the possessing iterations count
            # Print which team has possession, shot clock, and quarter

            clock_test = 'Quarter {:d} Time {:02d}:{:02d} Shot Clock {:03.1f}'.format(
                         current_quarter,
                         int(seconds_left) % 3600 // 60,
                         int(seconds_left) % 60,
                         current_shot_clock)
            print(clock_test)

        else:
            possessing_iterations += 1
            if possessing_iterations > 5:
                # If the same team has possession for more than 5 iterations, print it

                clock_test = 'Quarter {:d} Time {:02d}:{:02d} Shot Clock {:03.1f}'.format(
                         current_quarter,
                         int(seconds_left) % 3600 // 60,
                         int(seconds_left) % 60,
                         current_shot_clock)
                
                # print(f"{clock_test}, Possessing Team: {current_possessing_team}")

        # Stop checking possession after the first 5 iterations
        if iteration_counter >= 5:
            break

print(f"Total Number of Possessions: {num_possessions}")
