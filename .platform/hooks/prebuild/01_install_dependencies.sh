#!/bin/bash
set -xe

# Update all packages
sudo dnf update -y

# Install essential system dependencies for scientific Python libraries
sudo dnf install -y gcc gcc-c++ make \
    atlas-devel lapack-devel blas-devel \
    python3-devel

# Upgrade pip and setuptools
python3 -m pip install --upgrade pip setuptools wheel

# Install Python requirements
pip3 install -r requirements.txt
