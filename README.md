
# Real-time Upper body Measurement

## Project Overview
This project offers a real-time solution for upper body measurement and how much portion of the Upper body is accurately identified and detected by giving output in the form of confidence score between 0 to 1. It leverages the power of MediaPipe and OpenCV to accurately measure various body parts such as hips, shoulders, upper body. This tool can be used in various domains including fitness, fashion, and health monitoring.

## Features
- Real-time capturing of body measurements using a webcam.
- Utilization of MediaPipe for accurate upper body estimation.
- Calculation of distances between various body landmarks.
- Support for multiple persons' measurements.
- Data storage for both input and calculated measurements.

## Installation

### Prerequisites
- Python 3.11
- OpenCV
- MediaPipe
- A webcam or a video input device



## How It Works
- The application captures video from a webcam and uses MediaPipe to identify body landmarks.
- Measurements between specific landmarks (hips, shoulders, etc.) are calculated.
- Users are prompted to input real measurements for comparison purposes.


