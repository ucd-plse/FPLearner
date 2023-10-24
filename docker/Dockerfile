FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04

# Set working directory
RUN mkdir /root/home
WORKDIR /root/home

# Install OpenJdk Java 11 SDK
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get -y install openjdk-11-jdk-headless && rm -rf /var/lib/apt

# Install python3 and pip3
RUN apt-get update && apt-get -y install \
        python3 \
        python3-pip

# Install requirements
RUN apt-get update && apt-get -y install \
	vim \
	git \
	curl \
	unzip \
	zip \
	sudo

# Build llvm
COPY build_llvm_env.sh /root/
RUN chmod +x /root/build_llvm_env.sh
RUN /root/build_llvm_env.sh

# Install the release v1.1.922 of Joern
COPY install_joern.sh /root/
RUN chmod +x /root/install_joern.sh
RUN /root/install_joern.sh

# Install cmake and mpi for lulesh running
RUN sudo apt update && \
		sudo apt install -y software-properties-common lsb-release && \
		sudo apt clean all

RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null

RUN sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"

RUN sudo apt update && sudo apt install -y cmake && sudo apt install -y mpich

# pip install via requirements.txt
COPY requirements1.txt /root/
COPY requirements2.txt /root/
RUN pip3 install -r /root/requirements1.txt --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip3 install -r /root/requirements2.txt -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
