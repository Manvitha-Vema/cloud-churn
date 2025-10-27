#!/bin/bash
set -e

# Install system-level dependencies needed for numpy, pandas, and scipy
dnf install -y gcc gcc-c++ make atlas atlas-devel blas-devel lapack-devel python3-devel
