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



def train(inputStartIndex, inputEndIndex):

    folder_path_with_7z = r"D:\coding\NBA-Player-Movements\data\2016.NBA.Raw.SportVU.Game.Logs"
    destination_folder = r"D:\coding\GoogleCSR-Project\Dataset"

    batch_size = 5  # Number of files to process in each batch
    counter = 0

    sevenz_files = [filename for filename in os.listdir(folder_path_with_7z) if filename.endswith('.7z')][inputStartIndex-1:inputEndIndex]

    if not sevenz_files:
        exit()

    while counter < len(sevenz_files):
        

        # Process files in batches
        startIndex = counter
        endIndex = startIndex + batch_size

        if endIndex >= len(sevenz_files):
            endIndex = len(sevenz_files)

        for filename in sevenz_files:
            file_path = os.path.join(folder_path_with_7z, filename)

            with py7zr.SevenZipFile(file_path, mode='r') as archive:
                archive.extractall(destination_folder)
                print(f"Extracted {filename} to {destination_folder}")

            counter += 1


        delete_files_in_folder(destination_folder)

        extracted_files = os.listdir(destination_folder)
        print(f"Extracted files in {destination_folder}: {', '.join(extracted_files)}")
        print(f"Deleted {destination_folder} to save storage space")





        if endIndex == len(sevenz_files) - 1:
            break

    print(f"Processed {counter} .7z files in folder: {folder_path_with_7z}")




train(1,5)

train(630,633)



# NOTES
# Trained : 0 -- 5 (CHA at TOR) -- (PHI at LAL)
# Next game to train : 6 (BKN at BOS)