import json
import csv

# Define the path to your JSON file
json_file_path = r"C:\Users\rayya\OneDrive\Desktop\GoogleCSR-Project\Data_Preprocessing\PossessionTracker\0021500524.json"

# Open and read the JSON file
with open(json_file_path, "r") as json_file:
    data = json.load(json_file)

# Create a CSV file for storing the data
csv_file_path = "test.csv"

# Initialize the CSV writer
with open(csv_file_path, "w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write a header row to the CSV file
    csv_writer.writerow(["Quarter", "Time", "Shot Clock"])

    # Iterate through each event
    for event in data["events"]:
        # Iterate through each moment in the event
        for moment in event["moments"]:
            # Extract the seconds left in the current quarter, quarter number, and shot clock time
            current_time_since_1970 = moment[1]
            seconds_left = moment[2]
            current_quarter = moment[0]
            current_shot_clock = moment[3]

            if current_quarter is None or current_shot_clock is None or seconds_left is None:
                continue

            # Format the data as a string
            formatted_data = [
                current_quarter,
                '{:02d}:{:02d}'.format(int(seconds_left) % 3600 // 60, int(seconds_left) % 60),
                '{:03.1f}'.format(current_shot_clock)
            ]

            # Write the formatted data to the CSV file
            csv_writer.writerow(formatted_data)

print(f"Data has been saved to {csv_file_path}")
