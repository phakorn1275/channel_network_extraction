  # import image from docker hub
FROM amd64/python:3.9

# point working directory in docker image
WORKDIR /usr/src/app

# copy files from local pc to docker
COPY main.py .
COPY core.py .
# inside requirements are Python libraries that we want to use 
COPY requirements.txt .

# use pip to install all requirements
RUN pip install -r requirements.txt

# update pip
RUN python -m pip install -U pip

# install vim 
RUN apt-get update \
    && apt-get install -y \
        nmap \
        vim \
        unzip \
        wget

# trick to import subfolder
ADD image_out.zip .
ADD input_images.zip .