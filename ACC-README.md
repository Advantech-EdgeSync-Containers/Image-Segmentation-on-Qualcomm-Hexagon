# Image Segmentation on Qualcomm® Hexagon™

### About Advantech Container Catalog
The **Advantech Container Catalog** delivers hardware-accelerated AI containers pre-integrated for seamless edge deployment. These containers abstract complexities like SDK setup, runtime compatibility, and toolchain dependencies—offering rapid development pathways for platforms such as the **Qualcomm® QCS6490** SoC.

### Key benefits of the Container Catalog include:
| Feature / Benefit                | Description                                                                     |
| -------------------------------- | ------------------------------------------------------------------------------- |
| Optimized for Image Segmentation | Full stack for semantic segmentation with real-time video inference support     |
| Dual Export Workflows            | Supports both Qualcomm® AI Hub and Ultralytics-based model conversion pipelines |
| DSP & GPU Acceleration           | Utilizes Hexagon™ DSP 770 and Adreno™ 643 GPU for low-latency inference         |
| Multiple Runtime Support         | Integrated support for QNN, SNPE, and LiteRT runtimes                           |
| Format Flexibility               | Compatible with `.tflite`, `.dlc`, and `.so` models                             |
| Real-Time Vision Pipeline        | GStreamer-based multimedia framework with OpenCV integration                    |
| Fully Scripted Deployment        | Includes automated scripts for model export, quantization, and benchmarking     |
| ROS Integration                  | Compatible with Qualcomm Robotics Reference Distro with ROS 1.3-ver.1.1         |
| Versatile Use Cases              | Tailored for robotics, smart surveillance, healthcare, automotive, and more     |

## Container Overview

**Image Segmentation on Qualcomm® Hexagon™** is a comprehensive container solution for running real-time segmentation models on the **QCS6490** platform. Designed with **full DSP acceleration**, it brings plug-and-play deployment for models like **YOLOv8-seg** and **DeepLabv3+ MobileNet**, pre-optimized for edge scenarios.

This container offers:

* **Dual Image Segmentation Workflows**:

  * **Ultralytics Export**: Use YOLOv8-native tools to export to TFLite for rapid prototyping
  * **AI Hub Conversion**: Import optimized DeepLabv3+ MobileNet models directly from Qualcomm’s Hugging Face repository

* **Integrated Runtime Stack**:

  * Pre-installed support for **QNN**, **SNPE**, and **LiteRT**
  * Includes **GStreamer**, **OpenCV**, and **Python 3.10** for full inference pipeline development

* **Hardware-Accelerated Inference**:

  * INT8 inference on **Hexagon™ DSP 770**
  * FP32 fallback and GPU acceleration via **Adreno™ 643 GPU**

* **Multi-Model Format Compatibility**:

  * Runs **.tflite**, **.dlc**, and **.so** formats natively with supported runtimes

* **Preconfigured Scripts & Utilities**:

  * `advantech-coe-model-export.sh` and `advantech-aihub-model-export.sh` for model conversion
  * `wise-bench.sh` for validating runtime and AI environment

* **Ready for Industrial Edge Use Cases**:

  * Built for robotics, medical imaging, automotive vision, industrial inspection, and smart agriculture
  * Designed for use on **Advantech AOM-2721** with **QCS6490 SoC**

* **Seamless ROS Support**:

  * Compatible with **Qualcomm Robotics Reference Distro with ROS 1.3-ver.1.1** for plug-and-play robotic integration

## Container Demo
![Demo](%2Fdata%2Fgifs%2Fqc-yolo-seg-demo.gif)

## Use Cases

1. **Fitness & Rehabilitation**

   - Real-time exercise feedback: AI observe and correct posture during workouts for form optimization and injury prevention.
   - Physical therapy training: Monitor patient movement and progression during rehab, enabling remote guidance.

2. **Automotive & Robotics**

   - Autonomous navigation: Segment road surfaces, lanes, pedestrians, vehicles, and traffic signs in real time to enable accurate path planning and obstacle avoidance.
   - Robotic vision: Robots use scene-level segmentation to distinguish objects and environments, supporting tasks like pick-and-place, obstacle detection, and smooth human–robot interaction.

3. **Healthcare & Medical Imaging**

   - Tumor and organ segmentation: Precisely isolate tumors, organs, and anatomical structures from CT, MRI, and X-ray images to improve diagnostics, surgical planning, and treatment monitoring.
   - Quantitative analysis: Measure tissue volumes or morphological changes over time for progression tracking and intervention assessment.

4. **Satellite Imagery & Environmental Monitoring**

   - Land cover segmentation: Differentiate forest, water, urban, and agricultural regions in satellite images for land use classification and urban planning.
   - Disaster response & climate monitoring: Detect changes due to floods, deforestation, or shoreline erosion to support rapid decision-making and environmental protection.

5. **Smart Agriculture & Precision Farming**

   - Crop and plant health monitoring: Segment healthy vs. diseased crops and estimate yield using drone or satellite imagery to drive targeted interventions and reduce waste.
   - Weed detection: Separate weeds from crops to support precise herbicide application and bolster sustainable farming practices.

6. **Industrial Inspection & Quality Control**

   - Defect detection: Automatically identify scratches, cracks, or missing components on parts or PCBs in manufacturing pipelines to enable faster, more consistent quality checks.

7. **Retail, eCommerce & AR Experiences**

   - Virtual try-on & product isolation: Use foreground-background segmentation for virtual fitting rooms, product catalog consistency, and creative AR filters in apps.
   - Visual search & background removal: Automatically isolate products for better search and seamless visual editing in eCommerce platforms.

8. **Photography & Augmented Reality**

   - Selective editing & live filters: Enable portrait mode, background swap, or object removal with pixel-precise segmentation (e.g., Meta’s Segment Anything).

9. **Bio-Imaging & Research Applications**

   - Cell and subcellular segmentation: Segment cells, nuclei, or organelles in high-throughput microscopy for single-cell analysis, gene expression profiling, or drug discovery.

10. **Marine Ecology & Environmental Science**

    - Coral reef monitoring: Use segmentation tools like *TagLab* to quantify coral bleaching and monitor reef health through aerial or underwater imagery.
    - Shoreline mapping: Precisely segment the land-water boundary for erosion tracking, habitat assessment, and coastal planning.

## Key Features

- **Complete AI Framework Stack:** QNN SDK (QNN, SNPE), LiteRT

- **Edge AI Capabilities:** Optimized pipelines for real-time vision tasks (image segmentation)

- **Preconfigured Environment:** Comes with all necessary tools pre-installed in a container

- **Full DSP/GPU Acceleration:** Utilize Qualcomm® Hexagon™ DSP and Adreno™ GPU for fast and efficient inference

- **Dual Image Segmentation Workflows:** Support for both Qualcomm® AI Hub conversion and Ultralytics export methods, enabling better flexibility

## Host Device Prerequisites

| Component       | Specification      |
|-----------------|--------------------|
| Target Hardware | [Advantech AOM-2721](https://www.advantech.com/en/products/a9f9c02c-f4d2-4bb8-9527-51fbd402deea/aom-2721/mod_f2ab9bc8-c96e-4ced-9648-7fce99a0e24a) |
| SoC             | [Qualcomm® QCS6490](https://www.advantech.com/en/products/risc_evaluation_kit/aom-dk2721/mod_0e561ece-295c-4039-a545-68f8ded469a8)   |
| GPU             | Adreno™ 643        |
| DSP             | Hexagon™ 770       |
| Memory          | 8GB LPDDR5         |
| Host OS         | QCOM Robotics Reference Distro with ROS 1.3-ver.1.1       |


## Container Environment Overview

### Software Components on Container Image

| Component   | Version | Description                                                                                  |
|-------------|---------|----------------------------------------------------------------------------------------------|
| LiteRT      | 1.3.0   | Provides QNN TFLite Delegate support for GPU and DSP acceleration                            |
| [SNPE](https://docs.qualcomm.com/bundle/publicresource/topics/80-70014-15B/snpe.html)        | 2.29.0  | Qualcomm’s Snapdragon Neural Processing Engine; optimized runtime for Snapdragon DSP/HTP     |
| [QNN](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/introduction.html)         | 2.29.0  | Qualcomm® Neural Network (QNN) runtime for executing quantized neural networks                |
| GStreamer   | 1.20.7  | Multimedia framework for building flexible audio/video pipelines                             |
| Python   | 3.10.12  | Python runtime for building applications                             |
| OpenCV    | 4.11.0 | Computer vision library for image and video processing |


### Container Quick Start Guide
For container quick start, including the docker-compose file and more, please refer to [README.](https://github.com/Advantech-EdgeSync-Containers/Image-Segmentation-on-Qualcomm-Hexagon/blob/main/README.md)

### Supported AI Capabilities

#### Vision Models

| Model                               | Format       | Note                                                                 |
|-------------------------------------|--------------|----------------------------------------------------------------------|
| YOLOv8 Detection                    | TFLite INT8  | Downloaded from Ultralytics` official source and exported to TFLite using Ultralytics Python packages |
| YOLOv8 Segmentation                 | TFLite INT8  | Downloaded from Ultralytics` official source and exported to TFLite using Ultralytics Python packages |
| YOLOv8 Pose Estimation              | TFLite INT8  | Downloaded from Ultralytics` official source and exported to TFLite using Ultralytics Python packages |
| Lightweight Face Detector           | TFLite INT8  | Converted using Qualcomm® AI Hub                                       |
| FaceMap 3D Morphable Model          | TFLite INT8  | Converted using Qualcomm® AI Hub                                       |
| DeepLabV3+ (MobileNet)              | TFLite INT8  | Converted using Qualcomm® AI Hub                                       |
| DeepLabV3 (ResNet50)                | SNPE DLC TFLite | Converted using Qualcomm® AI Hub                                       |
| HRNet Pose Estimation (INT8)        | TFLite INT8  | Converted using Qualcomm® AI Hub                                       |
| PoseNet (MobileNet V1)              | TFLite       | Converted using Qualcomm® AI Hub                                       |
| MiDaS Depth Estimation              | TFLite INT8  | Converted using Qualcomm® AI Hub                                       |
| MobileNet V2 (Quantized)            | TFLite INT8  | Converted using Qualcomm® AI Hub                                       |
| Inception V3 (SNPE DLC)             | SNPE DLC TFLite | Converted using Qualcomm® AI Hub                                       |
| YAMNet (Audio Classification)       | TFLite       | Converted using Qualcomm® AI Hub                                       |
| YOLO (Quantized)                    | TFLite INT8  | Converted using Qualcomm® AI Hub                                       |

### Language Models Recommendation

| Model                               | Format       |   Note                                                         |
|-------------------------------------|--------------|----------------------------------------------------------------|
| Phi2                                | .so          | Converted using Qualcomm's LLM Notebook for Phi-2              |
| Tinyllama                           | .so          | Converted using Qualcomm's LLM Notebook for Tinyllama          |
| Meta Llama 3.2 1B                   | .so          | Converted using Qualcomm's LLM Notebook for Meta Llama 3.2 1B  |                                   |

## Supported AI Model Formats

| Runtime | Format  | Compatible Versions | 
|---------|---------|---------------------|
| QNN     | .so     |       2.29.0        |
| SNPE    | .dlc    |       2.29.0        |
| LiteRT  | .tflite |       1.3.0         | 

## Hardware Acceleration Support

| Accelerator | Support Level | Compatible Libraries |
|-------------|---------------|----------------------|
| GPU         |  FP32         | QNN, SNPE, LiteRT    |             
| DSP         |  INT8         | QNN, SNPE, LiteRT    |   

## Best Practices

* Prefer **INT8 quantized** models for DSP acceleration
* Ensure **fixed batch sizes** when converting models
* Use lower `GST_DEBUG` levels for stable multimedia handling
* Always validate exported models on-device after deployment