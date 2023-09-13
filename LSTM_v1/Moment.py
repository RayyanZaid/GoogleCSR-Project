# A moment looks like this : [x1,y1,x2,y2, .... x5,y5, bx,by,bz, shot_clock, game_clock]  :  LABEL

# Each moment should have :
    # 1) 25-D array with all the necessary spatio-temporal data
    # 2) A label

# Line from CSV : "[{'x': 47.68579, 'y': 24.46424}, {'x': 68.18295, 'y': 30.4235}, {'x': 56.14942, 'y': 18.65256}, {'x': 47.82405, 'y': 16.97243}, {'x': 50.9952, 'y': 32.64415}, {'x': 45.49446, 'y': 23.49714}, {'x': 45.63887, 'y': 34.14831}, {'x': 23.56491, 'y': 24.45822}, {'x': 44.05126, 'y': 17.08503}, {'x': 36.96702, 'y': 25.53479}]","{'x': 39.59473, 'y': 26.53377, 'z': 8.98725}",24.0,719.97,null

class Moment:

    def __init__(self, lineFromCSV):    
        self.lineFromCSV = lineFromCSV
        self.momentArray = []
        self.momentLabel :str = 0  # Initialize the label to 0

    def fillMomentInfoFromCSVLine(self):
        import json

        # Split the CSV line by comma to separate the values
        values = self.lineFromCSV.split(',')

        # Parse the spatial data from the first element (a JSON string)
        spatial_data = json.loads(values[0])

        # Extract the 'x' and 'y' values from the spatial data and append to momentArray
        for point in spatial_data:
            x_value = point['x']
            y_value = point['y']
            self.momentArray.extend([x_value, y_value])

        # Parse the 'z' value from the second element (a JSON string)
        z_data = json.loads(values[1])
        z_value = z_data.get('z', 0.0)  # Default to 0.0 if 'z' is missing
        self.momentArray.append(z_value)

        # Extract and assign the temporal values
        shot_clock = float(values[2])
        game_clock = float(values[3])

        # Populate the momentArray with the temporal values
        self.momentArray.extend([shot_clock, game_clock])

        # Extract and assign the label from the last column
        try:
            self.momentLabel = int(values[-1])
        except ValueError:
            # Handle invalid label value here
            self.momentLabel = 0  # Default to 0 for invalid labels


# FIX LATER
