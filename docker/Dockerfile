FROM ubuntu:22.04

# debian setting
ENV DEBIAN_FRONTEND=noninteractive \
    DEBIAN_FRONTEND=teletype \
    DEBCONF_NOWARNINGS=yes
# python pyc setting
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Common Package
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https \
    apt-utils \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    jq \
    gnupg \
    libgl1-mesa-dev \
    libopencv-dev \
    python3-dev \
    python3-pip \
    python3-setuptools \
    sudo \
    wget \
    zip \
  && apt-get -y clean \
  && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK
RUN mkdir -p /usr/share/keyrings \
 && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
     | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
 && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" \
     > /etc/apt/sources.list.d/google-cloud-sdk.list \
 && apt-get update \
 && apt-get install -y google-cloud-sdk \
 && apt-get -y clean \
 && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip3 install -U pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /root

# Set up the Bash shell environment
RUN echo "export PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> /root/.bashrc

ENTRYPOINT []

CMD ["/bin/bash"]