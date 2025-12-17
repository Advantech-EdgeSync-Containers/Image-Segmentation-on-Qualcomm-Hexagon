# Object Detection on Qualcomm® Hexagon™

### About Advantech Container Catalog
The **Advantech Container Catalog** delivers pre-integrated, hardware-accelerated containers optimized for rapid AI development on edge platforms. With support for the **Qualcomm® QCS6490**, these containers simplify the AI deployment pipeline by abstracting runtime dependencies, SDK installations, and toolchain configurations—letting developers focus on building impactful applications.

### Key benefits of the Container Catalog include:
| Feature / Benefit            | Description                                                                           |
| ---------------------------- | ------------------------------------------------------------------------------------- |
| Hardware-Accelerated Edge AI | Leverages DSP/NPU acceleration via QNN, SNPE, and LiteRT for low-latency inference    |
| YOLOv8 Ready Out-of-the-Box  | Supports both Ultralytics and AI Hub export workflows for rapid deployment            |
| Dual Workflow Flexibility    | Toggle between rapid prototyping and optimized deployment with script-based workflows |
| Multi-Model Format Support   | Compatible with `.tflite`, `.dlc`, and `.so` formats across runtimes                  |
| End-to-End Tooling Included  | Comes with export, quantization, and benchmarking scripts                             |
| Real-Time Vision Pipeline    | GStreamer + OpenCV integration for responsive video inference                         |
| ROS-Compatible               | Designed for use with Qualcomm® Robotics Reference Distro + ROS                       |
| Edge-Ready Use Cases         | Ideal for surveillance, retail, smart cities, robotics, and automation                |


## Container Overview

**Object Detection on Qualcomm® Hexagon™** is a plug-and-play container solution designed for running **YOLOv8 object detection** models on the **Qualcomm® QCS6490** platform. Built with **full DSP acceleration** support, it integrates the **QNN SDK**, **SNPE**, and **LiteRT** runtimes in a fully preconfigured environment, offering real-time detection capabilities out of the box.

This container offers:

* **Dual YOLOv8 Export Workflows**:

  * **Ultralytics Export** for rapid model testing using TFLite format
  * **Qualcomm® AI Hub Conversion** for optimized, INT8-accelerated deployments

* **Integrated Runtime Stack**:

  * QNN, SNPE, and LiteRT for full DSP/GPU support
  * GStreamer and OpenCV preconfigured for pipeline development

* **Deployment-Ready Hardware Acceleration**:

  * Direct INT8 inference on **Hexagon™ DSP 770**
  * GPU acceleration via **Adreno™ 643** for supported workloads

* **Flexible Model Format Support**:

  * Compatible with **TFLite**, **SNPE DLC**, and **QNN .so** model libraries

* **Preloaded Scripts & Tools**:

  * `advantech-coe-model-export.sh` and `advantech-aihub-model-export.sh` for fast conversion
  * `wise-bench.sh` for runtime environment verification

* **Built for On-Device Applications**:

  * Optimized for smart surveillance, industrial automation, retail analytics, and robotics
  * Designed to run on the **Advantech AOM-2721** with **QCS6490 SoC**

* **Works Seamlessly with ROS 2.0**:

  * Compatible with the **Qualcomm Robotics Reference Distro with ROS 1.3-ver.1.1**

## Container Demo
![Demo](%2Fdata%2Fgifs%2Fqc-yolo-det-demo.gif)

## Use Cases

1. **Industrial Automation**
   - Detect defective products on assembly lines in real time.
   - Monitor safety zones to ensure operators stay clear of robotic arms.
   - Track movement of tools or machinery parts for predictive maintenance.

2. **Smart Retail**
   - Customer behavior analytics (counting foot traffic, dwell time).
   - Shelf monitoring for stock-out detection and planogram compliance.
   - Automated checkout by detecting products without barcodes.

3. **Intelligent Transportation**
   - Real-time vehicle and pedestrian detection at intersections.
   - License plate localization for traffic management.
   - Monitoring driver behavior and detecting distractions in cabins.

4. **Robotics and Drones**
   - Object detection for autonomous navigation in warehouses or factories.
   - Detecting obstacles, pallets, or delivery packages.
   - Drone-based inspection of infrastructure (solar panels, power lines, bridges).

5. **Smart City and Surveillance**
   - Crowd density estimation and anomaly detection in public spaces.
   - Vehicle detection for parking management.
   - Perimeter surveillance with alerts for unauthorized entry.

6. **Healthcare and Assistive Systems**
   - Detecting PPE compliance (masks, gloves, helmets).
   - Monitoring patient activity in elderly care environments.
   - Assisting visually impaired individuals by identifying nearby objects.

7. **Agriculture**
   - Crop and livestock monitoring through aerial imaging.
   - Detecting weeds, pests, or diseases in fields.
   - Counting harvested produce in real time for yield estimation.

8. **Edge AI Research & Development**
   - Benchmarking new models on Qualcomm® DSP/GPU accelerators.
   - Evaluating trade-offs between INT8 and FP32 inference on real workloads.
   - Building custom datasets and retraining YOLOv8 for domain-specific applications.

## Key Features

- **Complete AI Framework Stack:** QNN SDK (QNN, SNPE), LiteRT

- **Edge AI Capabilities:** Optimized pipelines for real-time vision tasks (object detection)

- **Preconfigured Environment:** Comes with all necessary tools pre-installed in a container

- **Full DSP/GPU Acceleration:** Utilize Qualcomm® Hexagon™ DSP and Adreno™ GPU for fast and efficient inference

- **Dual YOLOv8 Detection Workflows:** Support for both Qualcomm® AI Hub conversion and Ultralytics export methods, enabling better flexibility

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
For container quick start, including the docker-compose file and more, please refer to [README.](https://github.com/Advantech-EdgeSync-Containers/Nagarro-Container-Project/blob/main/Object-Detection-on-Qualcomm-Hexagon/README.md)

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