import re

# Input text
text = r"C:\Users\rayya\OneDrive\Desktop\GoogleCSR-Project\Data_Preprocessing\MomentPreprocessing\0021500524.json"

# Define a regular expression pattern to match the number part
pattern = r'\d+'

# Use re.search to find the first occurrence of the pattern in the text
match = re.search(pattern, text)

# Check if a match was found
if match:
    # Extract and print the matched number part
    number_part = match.group()
    print("Extracted number part:", number_part)
else:
    print("No number found in the text.")
