# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy your application code into the container
COPY . /app

# Create a virtual environment
RUN python -m venv venv

# Activate the virtual environment using cmd.exe
SHELL ["cmd", "/S", "/C"]
RUN venv\Scripts\activate.bat

# Install the Python packages from requirements.txt inside the virtual environment
RUN pip install -r requirements.txt

# Expose any necessary ports (if your application requires it)
EXPOSE 80

# Set environment variables (if needed)
ENV NAME World

# Define the command to run your application
CMD ["python", "LiveResults/main.py", "LiveResults/0021500497.json", "1"]
