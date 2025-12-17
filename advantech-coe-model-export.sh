#!/bin/bash

# Exit on error
set -e

# Check arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 model=<model_name_or_path> task=<detection|pose|segmentation>"
    exit 1
fi

# Parse arguments
eval "$1"  # sets $model
eval "$2"  # sets $task

# Define working and output directories
WORKDIR="./yolo_tmp"
OUTPUTDIR="../model"

# Map task to suffix
case "$task" in
    detection)
        suffix="det"
        ;;
    pose)
        suffix="pose"
        ;;
    segmentation)
        suffix="seg"
        ;;
    *)
        echo "Error: Unsupported task '$task'. Use detection, pose, or segmentation."
        exit 1
        ;;
esac

# Create working directory
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Export the model
echo "Exporting YOLO model: $model"
yolo export model="$model" format=tflite int8=True imgsz=640

# Find the full integer quantized model
quant_model_file=$(find . -type f -name "*full_integer_quant.tflite" | head -n 1)

if [ -z "$quant_model_file" ]; then
    echo "Error: No full integer quantized model found!"
    exit 1
fi

# Prepare final filename
base_model_name=$(basename "$model" .pt)
final_model_name="${base_model_name}_${suffix}.tflite"

# Create output directory if not exist
mkdir -p "$OUTPUTDIR"

# Rename the model file before moving
renamed_model="./$final_model_name"
cp "$quant_model_file" "$renamed_model"

# Move renamed model to output dir
mv "$renamed_model" "$OUTPUTDIR/"

echo "Moved and renamed model to: $OUTPUTDIR/$final_model_name"

# Cleanup: remove everything else from temp dir
cd ..
rm -rf "$WORKDIR"
echo "Cleaned up $WORKDIR"

echo "✅ Done. Existing models in $OUTPUTDIR are untouched."
