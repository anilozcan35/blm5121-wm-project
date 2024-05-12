FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install dependencies and Streamlit in one layer to prevent unnecessary caching
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
    # pip install --no-cache-dir streamlit

# Copy the content of the local src directory to the working directory
COPY . .

EXPOSE 8080

CMD [ "streamlit", "run", "streamlit_app.py"]