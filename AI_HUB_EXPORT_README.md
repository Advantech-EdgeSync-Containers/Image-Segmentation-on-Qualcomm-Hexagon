# README: Deploy CV Models on QCS6490

**Device:** Qualcomm® QCS6490
- **Source:** [Qualcomm® Developer Docs](https://docs.qualcomm.com/bundle/publicresource/topics/80-70018-50/download-model-and-label-files.html)

---

## Prerequisites

### On the Target Device (QCS6490)
- Wi-Fi connected
- SSH access enabled

### On the Linux Host Machine
- Python environment installed
- Internet access for downloading scripts and model files

---

## Step 1: Get Token From Qualcomm® AI Hub

### On the Linux host:

1. **Create a Qualcomm® AI Hub account:**  
   [https://aihub.qualcomm.com](https://aihub.qualcomm.com)

2. **Sign in and open account settings:**

   Click your profile icon in the top-right corner, then select **Settings**.

3. **Locate your API token:**

   Scroll down to the **API Token** section.

   ![Locate API Token](%2Fdata%2Fimages%2FAI_HUB_API_TOKEN.png)

4. **Copy the token** for use in the next step.

---

## Step 2: Export the YOLOv8n Model Using the AI Hub Script

This step guides you through exporting the quantized YOLOv8n model. Choose the appropriate setup based on your current environment:
### First-Time Setup (Clone Needed)
```bash
# Clone the project repository
git clone https://github.com/Advantech-EdgeSync-Containers/Nagarro-Container-Project.git

# Move into the proper directory
cd Nagarro-Container-Project/Object-Detection-on-Qualcomm-Hexagon
```
### Repository Already Cloned
```bash
# Navigate directly to the working directory
cd Nagarro-Container-Project/Object-Detection-on-Qualcomm-Hexagon
```

### Export Model Script Execution
```bash
# Make the export script executable
chmod +x advantech-aihub-model-export.sh

# Execute the script using your API token
./advantech-aihub-model-export.sh --api-token=<YOUR_API_TOKEN>
```

This exports the quantized YOLOv8n model to your Linux machine.

---

## Step 3: Transfer Model, Config, Labels, and Media into the QCS6490 device

Use `scp` to push the model,config file, labels and media to the QCS6490 device:

```
scp -r AI_HUB_DATA root@<IP address of QCS6490>:/home/root
```

---

## Step 4: Transfer the Model, Config, Labels & Media into the Container

Use the following command to copy all files from your local `AI_HUB_DATA` directory into the running container’s `/etc/` path:

```bash
docker cp /home/root/AI_HUB_DATA/. Object-Detection-on-Qualcomm-Hexagon:/etc/
```
---
## Done!

Your QCS6490 is now ready to run AI/ML sample applications using the YOLOv8n model.
