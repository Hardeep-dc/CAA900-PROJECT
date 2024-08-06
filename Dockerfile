# Use the official Ubuntu 20.04 image as the base image
FROM ubuntu:20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Update PATH environment variable
ENV PATH /opt/conda/bin:$PATH

# Create a non-root user (optional but recommended)
RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo

# Switch to the non-root user
USER docker

# Set the working directory
WORKDIR /home/docker

# Copy the requirements file into the container
COPY requirements.txt .

# Initialize Conda, create environment, and install dependencies
RUN /opt/conda/bin/conda init bash && \
    /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda create -y -n myenv python=3.8 && conda activate myenv && conda install -y --file requirements.txt && conda install -y -c conda-forge prophet"

# Copy the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Run the application
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate myenv && python app.15.py"]
