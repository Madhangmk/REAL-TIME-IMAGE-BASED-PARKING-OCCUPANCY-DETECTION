# Parking Free Space Detection Using Computer Vision

---

## 1. Title of the Project

**REAL TIME IMAGE BASED PARKING OCCUPANCY DETECTION USING AI**

---

## 2. Introduction about the Project

The Parking Free Space Detection Using Computer Vision project is an intelligent system designed to automatically monitor and detect available parking spaces in real-time video feeds. This project leverages state-of-the-art deep learning object detection models combined with polygon-based spatial analysis to accurately identify whether parking slots are occupied or vacant.

Traditional parking management systems rely on manual monitoring or expensive sensor infrastructure, making them costly to implement and maintain. This computer vision-based solution provides a cost-effective alternative by utilizing existing CCTV cameras or video footage to detect vehicles and determine parking space availability automatically.

The system is designed to work with various video sources including pre-recorded videos, live camera streams, and surveillance footage from parking lots. Users can define custom parking space polygons directly on the video frame, making the system adaptable to different parking lot layouts and configurations.

---

## 3. Software & Hardware Specification

### Software Specifications

| Component                   | Specification                               |
| --------------------------- | ------------------------------------------- |
| **Programming Language**    | Python 3.x                                  |
| **Framework**               | Flask (Web Application), Ultralytics YOLOv8 |
| **Computer Vision Library** | OpenCV (cv2) >= 4.9.0.80                    |
| **Deep Learning Framework** | PyTorch (via Ultralytics)                   |
| **Geometric Processing**    | Shapely >= 2.0.3                            |
| **Numerical Computing**     | NumPy >= 1.26.4                             |
| **Web Server**              | Flask with Werkzeug                         |
| **Database**                | SQLite3 (for user authentication)           |
| **Frontend**                | HTML5, CSS3, JavaScript                     |
| **Operating System**        | Windows, Linux, macOS                       |

### Hardware Specifications

| Component     | Minimum Requirement               | Recommended                                 |
| ------------- | --------------------------------- | ------------------------------------------- |
| **Processor** | Intel Core i3 / AMD Ryzen 3       | Intel Core i7 / AMD Ryzen 7                 |
| **RAM**       | 4 GB                              | 8 GB or higher                              |
| **Storage**   | 10 GB free space                  | 20 GB SSD                                   |
| **GPU**       | Not required (CPU mode supported) | NVIDIA GPU with CUDA (for faster inference) |
| **Display**   | 1280x720 resolution               | 1920x1080 or higher                         |

### Model Specifications

| Parameter                | Value                             |
| ------------------------ | --------------------------------- |
| **Detection Model**      | YOLOv8m (Medium)                  |
| **Training Epochs**      | 101                               |
| **Image Size**           | 750 pixels                        |
| **Input Resolution**     | 1280x720 (resized for processing) |
| **Confidence Threshold** | Default (model default)           |
| **IoU Threshold**        | 0.7                               |

---

## 4. Problem Identified / Project Objective

### Problems Identified

1. **Manual Parking Monitoring**: Traditional parking lots require personnel to manually monitor and guide drivers to available spaces, which is labor-intensive and prone to human error.

2. **High Infrastructure Cost**: Installing physical sensors or magnetic loops in each parking space is expensive and requires significant maintenance.

3. **Inefficient Parking Search**: Drivers often spend excessive time searching for available parking spaces, leading to frustration, wasted fuel, and increased traffic congestion.

4. **Lack of Real-time Information**: Many parking facilities do not provide real-time availability information to incoming drivers.

5. **Limited Scalability**: Existing sensor-based solutions are difficult to scale across large parking areas or multiple locations.

### Project Objectives

1. **Automate Parking Detection**: Develop an automated system that can detect available parking spaces without human intervention using computer vision.

2. **Real-time Processing**: Provide real-time or near-real-time detection of parking space availability from video feeds.

3. **User-friendly Interface**: Create an intuitive web-based interface that allows users to easily upload videos, define parking polygons, and view results.

4. **Multi-vehicle Detection**: Detect various types of vehicles including cars, trucks, buses, motorcycles, bicycles, and vans.

5. **Scalable Solution**: Design a system that can be easily scaled to cover large parking areas using existing CCTV infrastructure.

6. **Cost-effective Implementation**: Eliminate the need for expensive hardware sensors by utilizing computer vision and existing video sources.

---

## 5. Description about the Project and Technology Used

### Project Overview

The Parking Free Space Detection system consists of two main components:

1. **Standalone Desktop Application** - A direct video processing application using OpenCV
2. **Web Application** - A Flask-based web platform for user authentication, video upload, and interactive parking space management

### Technology Stack

#### Core Technologies

**YOLOv8 (You Only Look Once Version 8)**

- State-of-the-art object detection model from Ultralytics
- Uses a CSPDarknet backbone for feature extraction
- Delivers real-time inference with high accuracy
- Trained to detect 10 object classes: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, and motor

**OpenCV (Open Computer Vision)**

- Primary library for video frame processing
- Handles video capture, frame manipulation, and visualization
- Provides polygon drawing and masking capabilities

**Flask Web Framework**

- Lightweight Python web framework
- Handles HTTP requests, routing, and template rendering
- Provides RESTful API endpoints for video processing

**Shapely Geometry Library**

- Advanced geometric operations
- Polygon intersection calculations
- Point-in-polygon testing

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                     │
│  (Web: HTML/CSS/JS or Desktop: OpenCV Window)              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  (Flask App / Main Processing Script)                       │
│  - Video Upload & Frame Extraction                         │
│  - Polygon Management (Save/Load)                           │
│  - Authentication & Session Management                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Detection Layer                           │
│  (YOLOv8 Model Inference)                                  │
│  - Vehicle Detection & Classification                       │
│  - Bounding Box Generation                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Analysis Layer                             │
│  - Polygon Center Calculation                              │
│  - Point-in-Polygon Testing                                │
│  - Occupancy Determination                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Output Layer                               │
│  - Visual Overlay (Red: Occupied, Yellow: Free)            │
│  - Statistics Display (Total/Free/Occupied)                │
│  - JSON Data Export                                        │
└─────────────────────────────────────────────────────────────┘
```

### Detection Algorithm

1. **Frame Extraction**: Video frames are captured and resized to 1280x720 resolution.

2. **Vehicle Detection**: YOLOv8 model processes each frame and identifies vehicles with bounding boxes, confidence scores, and class labels.

3. **Polygon Definition**: Users define parking space polygons by clicking on the video frame. Each polygon represents a single parking space.

4. **Center Point Calculation**: The center point of each parking polygon is computed using the centroid formula.

5. **Occupancy Detection**: For each detected vehicle, the system checks if any parking polygon center point falls within the vehicle's bounding box.

6. **Visualization**:
   - Red overlay: Occupied parking spaces
   - Yellow overlay: Free parking spaces

7. **Statistics**: Real-time display of total spaces, free spaces, and occupied spaces.

### Web Application Features

- **User Authentication**: Registration and login system with SQLite database
- **Video Management**: Upload, store, and process video files
- **Interactive Polygon Editor**: Click-to-draw parking spaces on video frames
- **Real-time Processing**: Process individual frames and display results
- **Video Information**: Display total frames, FPS, and processing progress

---

## 6. Merits of the Project

### Technical Merits

1. **Real-time Performance**: YOLOv8 model provides fast inference, enabling near real-time parking detection.

2. **High Accuracy**: The custom-trained model achieves ~48% mAP (mean Average Precision), demonstrating reliable vehicle detection.

3. **Multi-class Detection**: Can detect 10 different types of objects including various vehicle categories.

4. **Flexible Polygon System**: Users can define custom parking spaces by drawing polygons, making the system adaptable to any parking layout.

5. **CPU-friendly**: Works efficiently on CPU, making it accessible without expensive GPU hardware.

6. **Modular Design**: Clear separation between detection, analysis, and visualization layers.

### Practical Merits

1. **Cost-effective**: Eliminates need for expensive sensor infrastructure.

2. **Easy Deployment**: Can use existing CCTV cameras without modification.

3. **User-friendly Interface**: Web application provides intuitive controls for non-technical users.

4. **Automated Process**: No manual intervention required once parking spaces are defined.

5. **Scalable**: Can be easily extended to monitor multiple camera feeds.

### Software Engineering Merits

1. **Open Source**: Built entirely on open-source libraries and frameworks.

2. **Cross-platform**: Works on Windows, Linux, and macOS.

3. **Well-structured Code**: Modular functions for polygon operations, detection, and video processing.

4. **Data Persistence**: Polygon configurations saved in JSON/OBJ format for reuse.

---

## 7. Limitations or Challenges in the Project

### Technical Limitations

1. **Occlusion Issues**: Vehicles partially hidden behind other objects may not be detected accurately.

2. **Lighting Conditions**: Performance degrades in poor lighting conditions (night, heavy shadows, glare).

3. **Camera Angle**: Detection accuracy depends on camera perspective; oblique angles may reduce precision.

4. **Processing Speed**: CPU-based processing may experience latency with high-resolution videos.

5. **Model Generalization**: The model is trained on specific datasets and may not detect all vehicle types or unusual parking configurations.

6. **Static Polygon Positions**: Parking spaces must be manually defined and cannot automatically adapt to camera angle changes.

### Implementation Challenges

1. **Complex Polygon Editing**: Drawing accurate polygons requires multiple clicks and may be time-consuming for large parking areas.

2. **Video Format Compatibility**: Limited to specific video formats (MP4, AVI, MOV, MKV).

3. **Storage Management**: Processed videos and polygon data require adequate storage capacity.

4. **Network Dependencies**: Web application requires stable network for video upload/download.

5. **Frame-by-frame Processing**: Current implementation processes individual frames rather than continuous video streaming.

### Environmental Challenges

1. **Weather Conditions**: Rain, snow, or fog may affect video quality and detection accuracy.

2. **Camera Maintenance**: Requires periodic camera calibration and maintenance.

3. **Background Clutter**: Busy backgrounds may cause false positives in detection.

---

## 8. Conclusion

The Parking Free Space Detection Using Computer Vision project successfully demonstrates an innovative approach to automated parking management. By combining YOLOv8 object detection with interactive polygon-based space definition, the system provides a practical solution for real-time parking monitoring.

The project achieves its primary objectives of:

- Automated vehicle detection using deep learning
- Real-time parking space occupancy analysis
- User-friendly web interface for easy operation
- Cost-effective implementation without specialized hardware

The system shows promising results in detecting various vehicle types and accurately determining parking space availability. The modular architecture allows for easy integration into existing parking management systems and can be extended to support multiple camera feeds.

The combination of standalone desktop application and web-based platform provides flexibility for different use cases - from individual parking lots to large-scale commercial parking facilities.

While the current implementation has certain limitations related to lighting conditions and camera angles, the project provides a solid foundation for further development and improvement.

---

## 9. Future Work

### Short-term Improvements

1. **Enhanced Model Training**: Train the YOLOv8 model on more diverse datasets to improve detection accuracy and generalization.

2. **GPU Acceleration**: Implement CUDA-based processing for faster real-time inference.

3. **Night Mode**: Add infrared/night vision support for 24/7 operation.

4. **Mobile Application**: Develop mobile apps (iOS/Android) for remote parking monitoring.

5. **Multi-camera Integration**: Support simultaneous processing of multiple camera feeds.

### Medium-term Enhancements

6. **Automatic Polygon Generation**: Implement AI-based automatic detection of parking space lines to auto-generate polygons.

7. **Prediction Analytics**: Add historical data analysis to predict parking availability patterns.

8. **Integration with Payment Systems**: Connect with parking payment platforms for automated billing.

9. **Mobile Notification System**: Send push notifications to users when parking becomes available.

10. **Voice Assistance**: Add voice commands for hands-free polygon editing and system control.

### Long-term Vision

11. **Smart City Integration**: Connect with city-wide traffic management systems for coordinated parking guidance.

12. **Autonomous Vehicle Support**: Provide APIs for autonomous vehicles to query parking availability.

13. **Machine Learning Updates**: Implement online learning to continuously improve detection based on feedback.

14. **3D Parking Mapping**: Use stereo vision or depth cameras for more accurate space detection.

15. **Cloud Deployment**: Deploy as a cloud service for scalable, distributed parking monitoring across multiple locations.

---

## Appendix: Technology Summary

| Category          | Technology          |
| ----------------- | ------------------- |
| **Language**      | Python 3.x          |
| **ML Framework**  | Ultralytics YOLOv8  |
| **CV Library**    | OpenCV              |
| **Web Framework** | Flask               |
| **Database**      | SQLite              |
| **Frontend**      | HTML/CSS/JavaScript |
| **Deployment**    | Local/Cloud         |

---

_Project Documentation Version 1.0_
_Created for Parking Free Space Detection Using Computer Vision_
