# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy your application code into the container
COPY . /app

RUN pip install -r requirements.txt



# Define the command to run your application
CMD ["python", "LiveResults/main.py", "LiveResults/0021500497.json", "1"]
