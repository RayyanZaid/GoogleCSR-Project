import os 
import py7zr


# print(py7zr.__version__)
# print(os.name)

folder_path_with_7z = r"D:\coding\NBA-Player-Movements\data\2016.NBA.Raw.SportVU.Game.Logs"

# print(folder_path_with_7z)

counter = 0

for filename in os.listdir(folder_path_with_7z):

    file_path = os.path.join(folder_path_with_7z, filename)

    counter += 1
    print(file_path)

    if filename.endswith('.7z'):
        with py7zr.SevenZipFile(file_path , mode = 'r') as archive:

            destination_folder = r"D:\coding\GoogleCSR-Project\Dataset"
            archive.extractall(destination_folder)
            print(f"Extracted {filename} to {destination_folder}")

    break

print(f"There are {counter} .7z files in folder : {folder_path_with_7z}")