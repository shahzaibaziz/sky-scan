# SkyScan

## Overview
SkyScan is an AI-driven project designed to analyze aerial imagery from the Sheffield City region. Utilizing advanced image segmentation techniques, SkyScan identifies and classifies building materials and textures—such as concrete, metal, glass, solar panels, and tiles—to provide actionable insights for urban planning and decision-making.

## Objectives
- Create a user-friendly platform for analyzing aerial imagery.
- Compare and evaluate the performance of various image segmentation algorithms.
- Ensure high accuracy, efficiency, and functionality in detecting building materials and textures.

## Deliverables
SkyScan delivers an AI-powered tool that:
- Processes satellite imagery to segment and identify buildings.
- Provides detailed reports on roof materials, colors, and textures.
- Supports Sheffield City Council with data-driven urban planning solutions.

## Requirements
- **Environment**: Cloud-based GPU machine running Python 3.8.
- **IDE**: PyCharm configured for remote server access via SSH.
- **Dependencies**: Specific library versions compatible with Python 3.8 (listed below).

## Setup and Installation

### Prerequisites
1. Access to a cloud GPU machine with Python 3.8 installed.
2. PyCharm configured for SSH connection to the remote server.
3. Jupyter server enabled on the remote machine.

### Required Libraries
The following libraries are required and compatible with Python 3.8:
- tensorflow==2.9.1
- torch==1.12.1
- torchvision==0.13.1
- opencv-python==4.6.0.66
- scikit-image==0.19.3
- numpy==1.21.6
- pandas==1.3.5
- matplotlib==3.5.3
- seaborn==0.11.2
- scikit-learn==1.0.2
- jupyter==1.0.0

### Installation
Install the required libraries using the following commands:
```bash
pip install tensorflow==2.9.1
pip install torch==1.12.1 torchvision==0.13.1
pip install opencv-python==4.6.0.66
pip install scikit-image==0.19.3
pip install numpy==1.21.6
pip install pandas==1.3.5
pip install matplotlib==3.5.3
pip install seaborn==0.11.2
pip install scikit-learn==1.0.2
pip install jupyter==1.0.0
```

## Running the Project

1. **Upload Data**:
   - Use PyCharm’s deployment tools to transfer datasets and imagery to the remote server.
   - Verify that files are placed in the correct directory via SSH.

2. **Connect to the Server**:
   - Open PyCharm and connect to the remote Jupyter server.
   - Test the connection by running a simple Python script or notebook.

3. **Execute Playbooks**:
   - Open project playbooks in PyCharm.
   - Run playbooks individually or in batch mode using the PyCharm interface.

4. **Analyze Outputs**:
   - Use the Jupyter server in PyCharm to visualize and analyze results.

## Troubleshooting
- **Dependency Issues**: Ensure all libraries match the specified versions. Install missing dependencies using `pip`.
- **Connection Problems**: Verify SSH settings in PyCharm and check server status.
- **Logs**: Review Docker logs or PyCharm terminal output for errors.

## Notes
- The project is optimized for Python 3.8; avoid using incompatible library versions.
- Contact the project team for access to the cloud GPU machine or additional support.
