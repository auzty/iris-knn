# Base Image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy whole directory to /usr/src/app
COPY . /usr/src/app

# run pip
RUN pip install --no-cache-dir -r /usr/src/app/requirements.txt

# Command when Container is Run
CMD ["python","/usr/src/app/iris-test.py"]