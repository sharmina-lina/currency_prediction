# Use the official Python image from the DockerHub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any required packages from the requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Set environment variables (optional)
# ENV SOME_ENV_VAR=value

# Expose port if your app runs on a specific port
# EXPOSE 8000

# Run the application (you can adjust this command according to your project)
CMD ["python", "prediction.py"]
