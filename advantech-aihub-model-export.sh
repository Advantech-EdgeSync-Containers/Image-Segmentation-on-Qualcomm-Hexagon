# Copyright (c) 2024 Qualcomm® Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

#!/bin/bash
set -e

# Install unzip on Ubuntu/Debian if not already available
if ! command -v unzip &>/dev/null; then
  echo "Installing unzip utility via apt..."
  sudo apt-get install -y unzip
fi

# Function to download and unzip files
download_models() {
    local url=$1
    local output_dir=$2
    curl -L -O "$url" && unzip -o "$(basename "$url")" && \
    mv "$(basename "$url" .zip)"/* "$output_dir"
    #Cross Check : should it be cp or mv cp "$(basename "$url" .zip)"/* "$output_dir"
    rm -rf $(basename "$url" .zip)
    rm -rf $(basename "$url")
}

download_labels() {
    local url=$1
    local output_dir=$2
    curl -L -O "$url" && unzip -o "$(basename "$url")" && \
    cp labels/yolonas.labels labels/yolov8.labels
    mv labels/* "$output_dir"
    #Cross Check : should it be cp or mv?  cp labels/* "$output_dir"
    rm -rf $(basename "$url" .zip)
    rm -rf $(basename "$url")
}

# Function to download files
download_file() {
    local url=$1
    local output_dir=$2
    curl -L -O "$url"
    mv "$(basename "$url")" "$output_dir"
    #Cross Check again: cp "$(basename "$url")" "$output_dir"
}

# Function to download configs
download_config() {
    local url=$1
    local output_dir=$2
    curl -L -o "$output_dir" "$url"
}

outputmodelpath="./AI_HUB_DATA/models/"
outputlabelpath="./AI_HUB_DATA/labels/"
outputconfigpath="./AI_HUB_DATA/configs/"
outputmediapath="./AI_HUB_DATA/media/"

mkdir -p "${outputmodelpath}"
mkdir -p "${outputlabelpath}"
mkdir -p "${outputconfigpath}"
mkdir -p "${outputmediapath}"
 
download_labels "https://github.com/quic/sample-apps-for-qualcomm-linux/releases/download/GA1.3-rel/labels.zip" ${outputlabelpath}
download_file "https://raw.githubusercontent.com/quic/sample-apps-for-qualcomm-linux/refs/heads/main/artifacts/videos/video.mp4" "${outputmediapath}/"
download_file "https://raw.githubusercontent.com/quic/sample-apps-for-qualcomm-linux/refs/heads/main/artifacts/videos/video1.mp4" "${outputmediapath}/"
download_file "https://raw.githubusercontent.com/quic/sample-apps-for-qualcomm-linux/refs/heads/main/artifacts/videos/video-flac.mp4" "${outputmediapath}/"
download_file "https://raw.githubusercontent.com/quic/sample-apps-for-qualcomm-linux/refs/heads/main/artifacts/videos/video-mp3.mp4" "${outputmediapath}/"
download_config "https://git.codelinaro.org/clo/le/platform/vendor/qcom-opensource/gst-plugins-qti-oss/-/raw/imsdk.lnx.2.0.0.r2-rel/gst-sample-apps/gst-ai-object-detection/config_detection.json?inline=false" "${outputconfigpath}/config_detection.json"

# Parse command line arguments for API token
for i in "$@"; do
  case $i in
    --api-token=*|--api-key=*)
      API_TOKEN="${i#*=}"
      shift
      ;;
    *)
      echo "Usage: $0 --api-token=your_api_token_here or --api-key=your_api_key_here"
      exit 1
      ;;
  esac
done

# Check if API token is provided
if [ -z "$API_TOKEN" ]; then
  echo "API token is required. Usage: $0 --api-token=your_api_token_here or --api-key=your_api_key_here"
  exit 1
fi

# Define the Miniconda installer URL
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh"

# Define the installation directory
INSTALL_DIR="$PWD/miniconda3"

# Define the home directory for aihub assets
export HOME=$PWD

# Download the Miniconda installer
echo "Downloading Miniconda installer..."
wget $MINICONDA_URL -O miniconda.sh

# Make the installer executable
chmod +x miniconda.sh

# Run the installer
echo "Installing Miniconda..."
./miniconda.sh -b -u -p $INSTALL_DIR

# Initialize Conda
echo "Initializing Conda..."
$INSTALL_DIR/bin/conda init

# Clean up
rm miniconda.sh

# Activate Conda
echo "Activating Conda..."
source $INSTALL_DIR/bin/activate

# Create a Python 3.10 environment
echo "Creating Python 3.10 environment..."
$INSTALL_DIR/bin/conda create -y -n py310 python=3.10

# Activate the Python 3.10 environment
echo "Activating Python 3.10 environment..."
source $INSTALL_DIR/bin/activate py310

# Install the qai_hub package
echo "Installing qai_hub package..."
pip install python-git torchmetrics==1.8.1
# pip install qai-hub
# pip install "qai-hub-models[yolov8-det]"

pip install qai-hub==0.30
pip install 'qai_hub_models[yolov8-det]==0.30.2'

# Configure qai_hub with API token
echo "Configuring qai_hub..."
qai-hub configure --api_token "$API_TOKEN"

# Export models
echo "Exporting YOLOv8 quantized model..."
python3 -m qai_hub_models.models.yolov8_det.export --quantize w8a8 --target-runtime=tflite --chipset="qualcomm-qcs6490-proxy"

echo "Miniconda installation, activation, Python 3.10 environment creation, and qai_hub package installation complete."

source $INSTALL_DIR/bin/deactivate
source $INSTALL_DIR/bin/deactivate

cp build/yolov8_det/yolov8_det.tflite "${outputmodelpath}"
 