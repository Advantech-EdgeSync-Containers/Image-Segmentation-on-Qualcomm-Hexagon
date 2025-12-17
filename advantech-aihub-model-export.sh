# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
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

outputmodelpath="/etc/models/"
outputlabelpath="/etc/labels/"
outputconfigpath="/etc/configs/"
outputmediapath="/etc/media/"

mkdir -p "${outputmodelpath}"
mkdir -p "${outputlabelpath}"
mkdir -p "${outputconfigpath}"
mkdir -p "${outputmediapath}"
 
download_labels "https://github.com/quic/sample-apps-for-qualcomm-linux/releases/download/GA1.3-rel/labels.zip" ${outputlabelpath}
download_file "https://raw.githubusercontent.com/quic/sample-apps-for-qualcomm-linux/refs/heads/main/artifacts/videos/video.mp4" "${outputmediapath}/"
download_file "https://raw.githubusercontent.com/quic/sample-apps-for-qualcomm-linux/refs/heads/main/artifacts/videos/video1.mp4" "${outputmediapath}/"
download_file "https://raw.githubusercontent.com/quic/sample-apps-for-qualcomm-linux/refs/heads/main/artifacts/videos/video-flac.mp4" "${outputmediapath}/"
download_file "https://raw.githubusercontent.com/quic/sample-apps-for-qualcomm-linux/refs/heads/main/artifacts/videos/video-mp3.mp4" "${outputmediapath}/"
download_file "https://huggingface.co/qualcomm/DeepLabV3-Plus-MobileNet/resolve/2751392b3ca5e6e8cd3316f4c62501aa17c268e8/DeepLabV3-Plus-MobileNet_w8a8.tflite" "${outputmodelpath}/deeplabv3_plus_mobilenet_quantized.tflite"
download_config "https://git.codelinaro.org/clo/le/platform/vendor/qcom-opensource/gst-plugins-qti-oss/-/raw/imsdk.lnx.2.0.0.r2-rel/gst-sample-apps/gst-ai-segmentation/config_segmentation.json?inline=false" "${outputconfigpath}/config_segmentation.json"
