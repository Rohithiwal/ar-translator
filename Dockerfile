# Use the official Python 3.9 image
FROM python:3.9

# Set the working directory to /code
WORKDIR /code

# Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application code
COPY . /code

# Create a folder for secrets (optional, but good practice)
# We will mount the secret at runtime, so no need to copy a file here.

# Command to run the application
# Note: We use port 7860, which is the default for Hugging Face Spaces
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]