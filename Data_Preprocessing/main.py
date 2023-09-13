import json

# Define the path to your JSON file
json_file_path = r"C:\Users\rayya\OneDrive\Desktop\GoogleCSR-Project\Data_Preprocessing\PossessionTracker\0021500524.json"

# Open and read the JSON file
with open(json_file_path, "r") as json_file:
    data = json.load(json_file)

# Iterate through each event and print the "eventId"
for event in data["events"]:
    event_id = event["eventId"]
    print(f"Event ID: {event_id}")


