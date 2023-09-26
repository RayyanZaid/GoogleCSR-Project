import os
import py7zr



def delete_files_in_folder(folder_path):
    try:
        # List all files in the folder
        files = os.listdir(folder_path)

        # Iterate through the files and delete them
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")

        print(f"All files in {folder_path} have been deleted.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")



# Define the path to the folder containing .7z files
folder_path_with_7z = r"D:\coding\NBA-Player-Movements\data\2016.NBA.Raw.SportVU.Game.Logs"

# Define the destination folder where you want to extract files
destination_folder = r"D:\coding\GoogleCSR-Project\Dataset"

# Initialize a counter to keep track of the number of files processed
counter = 0

# Loop until there are no more .7z files in the folder
while True:
    # Find .7z files in the folder
    sevenz_files = [filename for filename in os.listdir(folder_path_with_7z) if filename.endswith('.7z')]

    # If there are no more .7z files, break out of the loop
    if not sevenz_files:
        break

    # Process up to 5 .7z files at a time
    for filename in sevenz_files[:5]:
        file_path = os.path.join(folder_path_with_7z, filename)

        # Extract the .7z file to the destination folder
        with py7zr.SevenZipFile(file_path, mode='r') as archive:
            archive.extractall(destination_folder)
            print(f"Extracted {filename} to {destination_folder}")

        # Increment the counter
        counter += 1

        # Delete the extracted .7z file
    delete_files_in_folder(destination_folder)

    extracted_files = os.listdir(destination_folder)
    print(f"Extracted files in {destination_folder}: {', '.join(extracted_files)}")
    
    print(f"Deleted {destination_folder} to save storage space")

# Print the total number of .7z files processed
print(f"Processed {counter} .7z files in folder: {folder_path_with_7z}")
