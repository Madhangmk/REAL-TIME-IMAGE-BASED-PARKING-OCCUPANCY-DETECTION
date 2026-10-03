# Parking Free Space Detection Using Computer Vision

## 🚗 Real-time Image-based Parking Occupancy Detection Using AI

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-green.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red.svg)](https://ultralytics.com/)
[![Flask](https://img.shields.io/badge/Flask-Web-orange.svg)](https://flask.palletsprojects.com/)

The Parking Free Space Detection project uses computer vision and deep learning to automatically detect available parking spaces from video feeds. Define custom parking polygons and get real-time occupancy status!

## ✨ Features

- **YOLOv8 Vehicle Detection** (~48% mAP on custom dataset)
- **Interactive Polygon Editor** for parking spaces
- **Flask Web App** with user auth, video upload
- **Standalone Python Script** for desktop use
- **Multi-vehicle support** (cars, vans, trucks, bikes, etc.)
- **Real-time Processing** on CPU/GPU

## 📋 Quick Start (Local)

1. **Clone & Setup**

   ```
   git clone https://github.com/YOUR_USERNAME/Parking-Free-Space-Detection.git
   cd Parking-Free-Space-Detection
   python -m venv .venv
   .venv/Scripts/activate  # Windows
   pip install -r requirements.txt
   ```

2. **Web App**

   ```
   cd web_app
   pip install -r requirements.txt
   python app.py
   ```

   Visit `http://localhost:5000`

3. **Standalone**
   ```
   python main.py
   ```

## 🗺️ Usage

1. Upload video or use webcam
2. Draw parking polygons on frame
3. Process → See occupancy (Red: occupied, Yellow: free)

**Demo Videos**: Check `Media/` samples

## 📁 Project Structure

```
.
├── main.py                 # Standalone app
├── functions.py            # Core detection logic
├── web_app/                # Flask web interface
├── Models/                 # YOLOv8 weights (download separately)
├── requirements.txt
├── README.md
└── .gitignore
```

## 🤝 Contributing

1. Fork the repo
2. Create feature branch
3. Commit changes
4. Push & PR

## 📄 License

MIT License - see [LICENSE](LICENSE) (add one)

## Acknowledgments

- Ultralytics YOLOv8
- OpenCV Team
- Flask Contributors

**⭐ Star the repo if helpful!**

---

_Originally based on PROJECT_DOCUMENTATION.md_
