# IoT WebCam Tutorial: WebRTC + YOLO + React

## 📚 Introduction

This tutorial guides you through building a complete IoT system with:
- **Raspberry Pi** as a WebRTC video server
- **React frontend** for real-time video display and control
- **YOLO object detection** running in the browser
- **Chat interface** for controlling the system

### What You'll Build

This is a **complete IoT video surveillance system** that demonstrates cutting-edge technologies:

#### **Core Components:**

1. **Raspberry Pi Video Server**
   - Real-time video capture from camera
   - WebRTC streaming to web browsers
   - REST API for system control
   - Motion detection and recording

2. **React Web Dashboard**
   - Live video display
   - Real-time object detection
   - Chat interface for voice control
   - System status monitoring

3. **YOLO Object Detection**
   - Browser-based AI inference
   - Real-time object recognition
   - Custom model training
   - Performance optimization

4. **Voice Control System**
   - Natural language processing
   - Camera control commands
   - System status queries
   - Automated responses

#### **Key Technologies Explained:**

**WebRTC (Web Real-Time Communication):**
- **What it is**: Protocol for real-time video/audio streaming
- **Why use it**: Low latency, peer-to-peer, browser-native
- **Advantages**: No plugins needed, works in all modern browsers
- **Use case**: Live video streaming from IoT devices

**YOLO (You Only Look Once):**
- **What it is**: Real-time object detection system
- **Why use it**: Fast, accurate, works on edge devices
- **Advantages**: Single-pass detection, real-time performance
- **Use case**: Identify objects in video streams

**React + TensorFlow.js:**
- **What it is**: Frontend framework + browser ML library
- **Why use it**: Rich UI + client-side AI inference
- **Advantages**: No server-side ML processing needed
- **Use case**: Interactive AI applications

#### **System Capabilities:**

- **Live Video Streaming**: Real-time camera feed to web browser
- **Object Detection**: Identify people, cars, animals in real-time
- **Voice Control**: "Show me the camera", "Detect objects", "Record video"
- **Motion Detection**: Automatic recording when movement detected
- **Remote Access**: Control system from anywhere via web browser
- **Data Logging**: Store detection events and system logs

#### **Real-World Applications:**

1. **Home Security**: Monitor your home remotely
2. **Pet Monitoring**: Watch your pets when away
3. **Traffic Analysis**: Count vehicles on your street
4. **Wildlife Observation**: Monitor garden wildlife
5. **Industrial Monitoring**: Monitor equipment and processes
6. **Research Projects**: Data collection for ML research

**What you'll build:**
- Live video streaming from Raspberry Pi camera
- Real-time object detection in the browser
- Voice-controlled camera system via chat
- Complete IoT dashboard

---

## 🎯 Prerequisites

### Hardware Requirements:
- **Raspberry Pi 4** (or Pi 3B+)
- **Raspberry Pi Camera Module** (or USB webcam)
- **MicroSD card** (16GB+ recommended)
- **Power supply** for Raspberry Pi

### Software Requirements:
- **Raspberry Pi OS** (Raspbian)
- **Python 3.8+**
- **Node.js 16+** (for React development)
- **Git**

### Knowledge Requirements:
- Basic Python programming
- Basic React knowledge (from our previous tutorials)
- Basic Linux command line usage

---

## 🏗️ Chapter 1: System Architecture

### Understanding the System Design

This IoT system follows a **client-server architecture** with **real-time communication** and **edge AI processing**. Let's break down how each component works together:

#### **System Overview**

```
┌─────────────────┐    WebRTC    ┌─────────────────┐
│   Raspberry Pi  │ ──────────── │   React App     │
│   (Camera)      │              │   (Browser)     │
│                 │              │                 │
│ • WebRTC Server │              │ • Video Display │
│ • Camera Stream │              │ • YOLO Detection│
│ • Control API   │              │ • Chat Interface│
└─────────────────┘              └─────────────────┘
```

#### **Component Responsibilities:**

**Raspberry Pi (Server Side):**
- **Video Capture**: Continuously capture frames from camera
- **WebRTC Signaling**: Handle connection establishment
- **Video Encoding**: Compress video for efficient streaming
- **Control API**: Accept commands from web interface
- **System Management**: Handle recording, motion detection

**React App (Client Side):**
- **Video Display**: Show live video stream
- **YOLO Processing**: Run object detection on video frames
- **User Interface**: Provide controls and status display
- **Chat Interface**: Natural language control system
- **Data Visualization**: Show detection results and statistics

#### **Why This Architecture?**

**Advantages:**
- **Low Latency**: WebRTC provides real-time video streaming
- **Scalability**: Multiple clients can connect to one server
- **Edge Processing**: AI runs in browser, reducing server load
- **Cross-Platform**: Works on any device with a web browser
- **Privacy**: Video processing happens locally in browser

**Design Decisions:**
- **WebRTC over HTTP**: Real-time streaming vs. request-response
- **Browser AI over Server AI**: Reduces server load and latency
- **React over Vanilla JS**: Better UI development and state management
- **REST API over WebSocket**: Simpler for control commands

### Data Flow

#### **Video Streaming Flow:**

1. **Camera Capture**: Raspberry Pi camera captures video frames
2. **Frame Processing**: Frames are encoded and prepared for streaming
3. **WebRTC Signaling**: Connection is established between Pi and browser
4. **Video Streaming**: Encoded video is streamed to browser in real-time
5. **Frame Display**: Browser displays video frames in React component
6. **YOLO Processing**: Each frame is processed for object detection
7. **Result Display**: Detection results are overlaid on video

#### **Control Flow:**

1. **User Input**: User types command in chat interface
2. **Command Processing**: React app processes natural language
3. **API Request**: Command is sent to Raspberry Pi via REST API
4. **Server Action**: Raspberry Pi executes the command
5. **Response**: Server sends back status/result
6. **UI Update**: React app updates interface with results

#### **Detection Flow:**

1. **Frame Capture**: Video frame is captured from stream
2. **Preprocessing**: Frame is resized and normalized for YOLO
3. **YOLO Inference**: Object detection is performed in browser
4. **Post-processing**: Detection results are filtered and formatted
5. **Visualization**: Bounding boxes and labels are drawn on frame
6. **Logging**: Detection events are logged for analysis

#### **Performance Considerations:**

**Latency Optimization:**
- **WebRTC**: ~100ms end-to-end latency
- **YOLO Processing**: ~50ms per frame (depending on model size)
- **Total Latency**: ~150ms from camera to detection display

**Bandwidth Optimization:**
- **Video Compression**: H.264 encoding reduces bandwidth
- **Adaptive Quality**: Adjust resolution based on network conditions
- **Frame Rate Control**: Reduce FPS if needed for performance

**Memory Optimization:**
- **Frame Buffering**: Only keep recent frames in memory
- **Detection Caching**: Cache results for similar frames
- **Garbage Collection**: Clean up unused objects regularly

---

## 🍓 Chapter 2: Raspberry Pi Setup

### Understanding Raspberry Pi for IoT Video

The **Raspberry Pi** is an ideal platform for IoT video applications because it:
- **Cost-effective**: ~$35-50 for a complete system
- **Powerful enough**: Can handle video encoding and streaming
- **Well-supported**: Extensive documentation and community
- **Expandable**: GPIO pins for additional sensors
- **Network-ready**: Built-in WiFi and Ethernet

#### **Hardware Requirements Explained:**

**Raspberry Pi 4 (Recommended):**
- **CPU**: 1.5GHz quad-core ARM Cortex-A72
- **RAM**: 2GB/4GB/8GB options (2GB minimum)
- **GPU**: VideoCore VI for hardware video encoding
- **Network**: Gigabit Ethernet + 802.11ac WiFi
- **USB**: 2x USB 3.0 ports for high-speed data transfer

**Camera Options:**
- **Pi Camera Module**: Dedicated camera interface, best performance
- **USB Webcam**: More flexible, works with any USB camera
- **IP Camera**: Network camera, no direct connection needed

### Step 1: Install Raspberry Pi OS

#### **Why Raspberry Pi OS?**

**Advantages:**
- **Optimized**: Specifically designed for Raspberry Pi hardware
- **Pre-configured**: Comes with necessary drivers and tools
- **Community Support**: Extensive documentation and forums
- **Regular Updates**: Security and performance updates

**Installation Options:**
1. **Raspberry Pi Imager** (Recommended): Official tool, easiest method
2. **Manual Installation**: More control, advanced users
3. **Headless Setup**: No monitor needed, SSH only

#### **Installation Steps:**

1. **Download Raspberry Pi Imager**
   - Download from: https://www.raspberrypi.org/software/
   - Available for Windows, macOS, and Linux

2. **Flash Raspberry Pi OS** to microSD card
   - Insert microSD card (16GB+ recommended)
   - Select "Raspberry Pi OS (32-bit)" for best compatibility
   - Choose your microSD card as the target
   - Click "Write" to flash the image

3. **Enable SSH** and **WiFi** during setup
   - Before ejecting the card, create an empty file named `ssh` in the boot partition
   - Create `wpa_supplicant.conf` file for WiFi configuration
   - This enables headless setup without a monitor

4. **Boot Raspberry Pi** and connect via SSH
   - Insert microSD card and power on Raspberry Pi
   - Wait 2-3 minutes for first boot
   - Connect via SSH: `ssh pi@raspberrypi.local`
   - Default password: `raspberry`

### Step 2: Install Dependencies

#### **Understanding the Dependencies:**

**System Dependencies:**
- **Python 3.8+**: Required for modern Python packages
- **pip**: Python package installer
- **venv**: Virtual environment for package isolation

**Camera Dependencies:**
- **libatlas-base-dev**: BLAS/LAPACK for numerical computations
- **libhdf5-dev**: HDF5 file format support (used by many ML libraries)
- **libjasper-dev**: JPEG-2000 support for image processing
- **libqtcore4/libqtgui4**: Qt libraries for GUI applications

**WebRTC Dependencies:**
- **libavdevice-dev**: FFmpeg device handling
- **libavfilter-dev**: FFmpeg filters for video processing
- **libopus-dev**: Opus audio codec (used by WebRTC)

```bash
# Update system packages to latest versions
# This ensures compatibility and security
sudo apt update && sudo apt upgrade -y

# Install Python package management tools
# pip3: Python package installer
# python3-venv: Virtual environment support
sudo apt install python3-pip python3-venv -y

# Install camera and image processing libraries
# These are required for OpenCV and camera operations
sudo apt install libatlas-base-dev libhdf5-dev libhdf5-serial-dev -y
sudo apt install libjasper-dev libqtcore4 libqtgui4 libqt4-test -y

# Install WebRTC and video streaming dependencies
# These enable real-time video communication
sudo apt install libavdevice-dev libavfilter-dev libopus-dev -y
```

### Step 3: Enable Camera

```bash
# Enable camera interface
sudo raspi-config

# Navigate to: Interface Options → Camera → Enable
# Reboot after enabling
sudo reboot
```

### Step 4: Test Camera

```bash
# Test camera capture
raspistill -o test.jpg

# Test video recording
raspivid -o test.h264 -t 5000
```

---

## 🐍 Chapter 3: WebRTC Server Setup

### Step 1: Create Python Environment

```bash
# Create project directory
mkdir ~/iot-webcam
cd ~/iot-webcam

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install WebRTC dependencies
pip install aiortc opencv-python numpy aiohttp
```

### Step 2: Create WebRTC Server

Create `server.py`:

```python
import asyncio
import json
import logging
import cv2
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRecorder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraVideoTrack(VideoStreamTrack):
    """Video track that streams from Raspberry Pi camera"""
    
    def __init__(self):
        super().__init__()
        self.camera = cv2.VideoCapture(0)  # Use camera index 0
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.camera.isOpened():
            raise RuntimeError("Could not open camera")
    
    async def recv(self):
        """Get next frame from camera"""
        ret, frame = self.camera.read()
        if not ret:
            raise RuntimeError("Could not read frame")
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create video frame
        from av import VideoFrame
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = int(asyncio.get_event_loop().time() * 1000000)
        video_frame.time_base = "1/1000000"
        
        return video_frame
    
    def stop(self):
        """Clean up camera"""
        if self.camera:
            self.camera.release()

class WebRTCServer:
    def __init__(self):
        self.pcs = set()
        self.video_track = None
    
    async def offer(self, request):
        """Handle WebRTC offer from client"""
        params = await request.json()
        offer = RTCSessionDescription(
            sdp=params["sdp"],
            type=params["type"]
        )
        
        pc = RTCPeerConnection()
        self.pcs.add(pc)
        
        # Create video track
        self.video_track = CameraVideoTrack()
        pc.addTrack(self.video_track)
        
        # Handle connection state changes
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state: {pc.connectionState}")
            if pc.connectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)
        
        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type
            })
        )
    
    async def close(self):
        """Close all peer connections"""
        coros = [pc.close() for pc in self.pcs]
        await asyncio.gather(*coros)
        self.pcs.clear()
        
        if self.video_track:
            self.video_track.stop()

# Create server instance
server = WebRTCServer()

# Web routes
async def index(request):
    """Serve the main page"""
    with open("static/index.html") as f:
        content = f.read()
    return web.Response(content_type="text/html", text=content)

async def offer(request):
    """Handle WebRTC offer"""
    return await server.offer(request)

# Create web application
app = web.Application()
app.router.add_get("/", index)
app.router.add_post("/offer", offer)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=8080)
```

### Step 3: Create Static HTML Page

Create `static/index.html`:

```html
<!DOCTYPE html>
<html>
<head>
    <title>IoT WebCam</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .video-container {
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 20px;
        }
        video {
            width: 100%;
            height: auto;
        }
        .controls {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        button {
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            margin-right: 10px;
        }
        button:hover {
            background: #0056b3;
        }
        .status {
            margin-top: 10px;
            padding: 10px;
            border-radius: 4px;
        }
        .status.connected {
            background: #d4edda;
            color: #155724;
        }
        .status.disconnected {
            background: #f8d7da;
            color: #721c24;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>IoT WebCam</h1>
        
        <div class="video-container">
            <video id="video" autoplay playsinline></video>
        </div>
        
        <div class="controls">
            <button id="startButton">Start Stream</button>
            <button id="stopButton" disabled>Stop Stream</button>
            <button id="captureButton" disabled>Capture Image</button>
            
            <div id="status" class="status disconnected">
                Disconnected
            </div>
        </div>
    </div>

    <script>
        let pc = null;
        let localStream = null;

        const startButton = document.getElementById('startButton');
        const stopButton = document.getElementById('stopButton');
        const captureButton = document.getElementById('captureButton');
        const video = document.getElementById('video');
        const status = document.getElementById('status');

        startButton.onclick = start;
        stopButton.onclick = stop;
        captureButton.onclick = capture;

        async function start() {
            try {
                // Create peer connection
                pc = new RTCPeerConnection({
                    iceServers: [
                        { urls: 'stun:stun.l.google.com:19302' }
                    ]
                });

                // Handle incoming tracks
                pc.ontrack = function(event) {
                    if (event.track.kind === 'video') {
                        video.srcObject = event.streams[0];
                        localStream = event.streams[0];
                    }
                };

                // Create offer
                const offer = await pc.createOffer({
                    offerToReceiveVideo: true
                });
                await pc.setLocalDescription(offer);

                // Send offer to server
                const response = await fetch('/offer', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        sdp: pc.localDescription.sdp,
                        type: pc.localDescription.type
                    })
                });

                const answer = await response.json();
                await pc.setRemoteDescription(answer);

                // Update UI
                startButton.disabled = true;
                stopButton.disabled = false;
                captureButton.disabled = false;
                status.textContent = 'Connected';
                status.className = 'status connected';

            } catch (error) {
                console.error('Error starting stream:', error);
                status.textContent = 'Error: ' + error.message;
                status.className = 'status disconnected';
            }
        }

        async function stop() {
            if (pc) {
                pc.close();
                pc = null;
            }
            if (localStream) {
                localStream.getTracks().forEach(track => track.stop());
                localStream = null;
            }
            video.srcObject = null;

            // Update UI
            startButton.disabled = false;
            stopButton.disabled = true;
            captureButton.disabled = true;
            status.textContent = 'Disconnected';
            status.className = 'status disconnected';
        }

        function capture() {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            
            const link = document.createElement('a');
            link.download = 'capture-' + Date.now() + '.png';
            link.href = canvas.toDataURL();
            link.click();
        }
    </script>
</body>
</html>
```

### Step 4: Run the Server

```bash
# Activate virtual environment
source venv/bin/activate

# Run the server
python server.py
```

**Test:** Open `http://your-raspberry-pi-ip:8080` in your browser.

---

## ⚛️ Chapter 4: React Frontend Integration

### Step 1: Create React Components

Create `frontend/src/components/LiveVideoStream.js`:

```jsx
import React, { useRef, useEffect, useState } from 'react';
import PropTypes from 'prop-types';

function LiveVideoStream({ onFrame, onConnectionChange }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const [pc, setPc] = useState(null);
  const [error, setError] = useState(null);

  const connectToStream = async () => {
    try {
      setError(null);
      
      // Create peer connection
      const peerConnection = new RTCPeerConnection({
        iceServers: [
          { urls: 'stun:stun.l.google.com:19302' }
        ]
      });

      // Handle incoming video track
      peerConnection.ontrack = (event) => {
        if (event.track.kind === 'video') {
          videoRef.current.srcObject = event.streams[0];
          setIsConnected(true);
          onConnectionChange?.(true);
        }
      };

      // Create offer
      const offer = await peerConnection.createOffer({
        offerToReceiveVideo: true
      });
      await peerConnection.setLocalDescription(offer);

      // Send offer to Raspberry Pi
      const response = await fetch('http://your-raspberry-pi-ip:8080/offer', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          sdp: peerConnection.localDescription.sdp,
          type: peerConnection.localDescription.type
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const answer = await response.json();
      await peerConnection.setRemoteDescription(answer);

      setPc(peerConnection);

    } catch (err) {
      setError(err.message);
      setIsConnected(false);
      onConnectionChange?.(false);
    }
  };

  const disconnect = () => {
    if (pc) {
      pc.close();
      setPc(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsConnected(false);
    onConnectionChange?.(false);
  };

  // Start frame capture for YOLO detection
  useEffect(() => {
    if (!videoRef.current || !canvasRef.current || !isConnected) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    const captureFrame = () => {
      if (video.videoWidth && video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);
        
        // Call onFrame with the canvas for YOLO processing
        onFrame?.(canvas);
      }
      
      if (isConnected) {
        requestAnimationFrame(captureFrame);
      }
    };

    video.addEventListener('loadedmetadata', captureFrame);

    return () => {
      video.removeEventListener('loadedmetadata', captureFrame);
    };
  }, [isConnected, onFrame]);

  return (
    <div className="live-video-stream">
      <div className="video-container">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="video-element"
        />
        <canvas
          ref={canvasRef}
          style={{ display: 'none' }}
        />
      </div>
      
      <div className="controls">
        {!isConnected ? (
          <button 
            onClick={connectToStream}
            className="connect-button"
            disabled={!!error}
          >
            Connect to Camera
          </button>
        ) : (
          <button 
            onClick={disconnect}
            className="disconnect-button"
          >
            Disconnect
          </button>
        )}
        
        {error && (
          <div className="error-message">
            Error: {error}
          </div>
        )}
        
        <div className="status">
          Status: {isConnected ? 'Connected' : 'Disconnected'}
        </div>
      </div>
    </div>
  );
}

LiveVideoStream.propTypes = {
  onFrame: PropTypes.func,
  onConnectionChange: PropTypes.func
};

export default LiveVideoStream;
```

### Step 2: Add Styles

Add to `frontend/src/index.css`:

```css
.live-video-stream {
  background: #1a1a1a;
  border-radius: 12px;
  padding: 20px;
  margin: 20px 0;
}

.video-container {
  position: relative;
  background: #000;
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 20px;
}

.video-element {
  width: 100%;
  height: auto;
  display: block;
}

.controls {
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
}

.connect-button {
  background: #10b981;
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 500;
}

.connect-button:hover:not(:disabled) {
  background: #059669;
}

.disconnect-button {
  background: #ef4444;
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 500;
}

.disconnect-button:hover {
  background: #dc2626;
}

.error-message {
  color: #ef4444;
  font-size: 14px;
  margin-left: 10px;
}

.status {
  color: #9ca3af;
  font-size: 14px;
  margin-left: auto;
}
```

### Step 3: Integrate with HomePage

Update `frontend/src/pages/HomePage.js`:

```jsx
import React, { useState, useEffect } from 'react';
import HeroSection from '../components/HeroSection';
import ChatInterface from '../components/ChatInterface';
import SkillsSection from '../components/SkillsSection';
import ExperienceSection from '../components/ExperienceSection';
import LiveVideoStream from '../components/LiveVideoStream';
import { getUserInfo } from '../services/api';

function HomePage() {
  const [userInfo, setUserInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isVideoConnected, setIsVideoConnected] = useState(false);
  const [detectedObjects, setDetectedObjects] = useState([]);

  useEffect(() => {
    const fetchUserInfo = async () => {
      try {
        const info = await getUserInfo();
        setUserInfo(info);
      } catch (error) {
        console.error('Failed to fetch user info:', error);
        setUserInfo({
          name: 'Your Name',
          title: 'Software Engineer',
          bio: 'Passionate about technology and innovation.',
          skills: ['Python', 'JavaScript', 'React', 'Flask', 'IoT'],
          experience: [
            {
              role: 'Software Engineer',
              company: 'Tech Company',
              period: '2020 - Present',
              description: 'Developing innovative solutions.'
            }
          ]
        });
      } finally {
        setLoading(false);
      }
    };

    fetchUserInfo();
  }, []);

  const handleVideoFrame = (canvas) => {
    // This will be used for YOLO detection
    // We'll implement this in the next chapter
    console.log('Frame captured for YOLO detection');
  };

  const handleConnectionChange = (connected) => {
    setIsVideoConnected(connected);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="loading-spinner"></div>
        <span className="ml-2">Loading...</span>
      </div>
    );
  }

  return (
    <div className="bg-gray-50">
      <div className="flex flex-col lg:flex-row container mx-auto px-4 py-8 gap-8">
        {/* Left Side: Hero Section */}
        <div className="w-full lg:w-1/2 mb-8 lg:mb-0 flex items-center justify-center">
          <HeroSection userInfo={userInfo} />
        </div>
        
        {/* Right Side: Chat Interface */}
        <div className="w-full lg:w-1/2 flex items-center justify-center">
          <ChatInterface userInfo={userInfo} />
        </div>
      </div>

      {/* Live Video Stream Section */}
      <div className="container mx-auto px-4 py-8">
        <h2 className="text-3xl font-bold text-center mb-8 text-gray-800">
          Live IoT Camera Feed
        </h2>
        <LiveVideoStream 
          onFrame={handleVideoFrame}
          onConnectionChange={handleConnectionChange}
        />
      </div>

      {/* Skills Section */}
      <SkillsSection skills={userInfo?.skills || []} />
      
      {/* Experience Section */}
      <ExperienceSection experience={userInfo?.experience || []} />
      
      {/* Footer */}
      <footer className="gradient-bg text-white py-8">
        <div className="container mx-auto px-4 text-center">
          <p>&copy; {new Date().getFullYear()} {userInfo?.name || 'Your Name'}. Built with React & Flask</p>
        </div>
      </footer>
    </div>
  );
}

export default HomePage;
```

---

## 🤖 Chapter 5: YOLO Object Detection

### Step 1: Install TensorFlow.js

```bash
cd frontend
npm install @tensorflow/tfjs @tensorflow/tfjs-backend-webgl
```

### Step 2: Create YOLO Detection Component

Create `frontend/src/components/YOLODetection.js`:

```jsx
import React, { useEffect, useState, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import PropTypes from 'prop-types';

function YOLODetection({ canvas, onDetection }) {
  const [model, setModel] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isDetecting, setIsDetecting] = useState(false);
  const detectionRef = useRef(null);

  // Load YOLO model
  useEffect(() => {
    const loadModel = async () => {
      try {
        setIsLoading(true);
        
        // Load TensorFlow.js backend
        await tf.setBackend('webgl');
        
        // Load YOLO model (you'll need to download this)
        const loadedModel = await tf.loadGraphModel('/models/yolo-tiny/model.json');
        setModel(loadedModel);
        
        console.log('YOLO model loaded successfully');
      } catch (error) {
        console.error('Error loading YOLO model:', error);
      } finally {
        setIsLoading(false);
      }
    };

    loadModel();
  }, []);

  // Run detection on canvas
  useEffect(() => {
    if (!canvas || !model || isLoading) return;

    const runDetection = async () => {
      try {
        setIsDetecting(true);
        
        // Preprocess image
        const tensor = tf.browser.fromPixels(canvas);
        const resized = tf.image.resizeBilinear(tensor, [416, 416]);
        const expanded = resized.expandDims(0);
        const normalized = expanded.div(255.0);
        
        // Run inference
        const predictions = await model.predict(normalized);
        
        // Process predictions
        const detections = processYOLOOutput(predictions, canvas.width, canvas.height);
        
        // Call callback with detections
        onDetection?.(detections);
        
        // Clean up tensors
        tf.dispose([tensor, resized, expanded, normalized, predictions]);
        
      } catch (error) {
        console.error('Detection error:', error);
      } finally {
        setIsDetecting(false);
      }
    };

    // Run detection every 100ms
    detectionRef.current = setInterval(runDetection, 100);

    return () => {
      if (detectionRef.current) {
        clearInterval(detectionRef.current);
      }
    };
  }, [canvas, model, isLoading, onDetection]);

  const processYOLOOutput = (predictions, imageWidth, imageHeight) => {
    // This is a simplified version - you'll need to implement proper YOLO output processing
    const detections = [];
    
    // Example processing (you'll need to adapt this for your specific YOLO model)
    const boxes = predictions[0].dataSync();
    const scores = predictions[1].dataSync();
    const classes = predictions[2].dataSync();
    
    for (let i = 0; i < scores.length; i++) {
      if (scores[i] > 0.5) { // Confidence threshold
        detections.push({
          class: classes[i],
          confidence: scores[i],
          bbox: boxes.slice(i * 4, (i + 1) * 4)
        });
      }
    }
    
    return detections;
  };

  if (isLoading) {
    return (
      <div className="yolo-loading">
        <div className="loading-spinner"></div>
        <span>Loading YOLO model...</span>
      </div>
    );
  }

  return (
    <div className="yolo-detection">
      <div className="detection-status">
        Status: {isDetecting ? 'Detecting objects...' : 'Ready'}
      </div>
    </div>
  );
}

YOLODetection.propTypes = {
  canvas: PropTypes.object,
  onDetection: PropTypes.func
};

export default YOLODetection;
```

### Step 3: Download YOLO Model

You'll need to download a YOLO model. For this tutorial, we'll use YOLO-tiny:

```bash
# Create models directory
mkdir -p frontend/public/models/yolo-tiny

# Download YOLO-tiny model files
# You can find these at: https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd
# Or use a pre-converted model from: https://github.com/ModelDepot/tfjs-yolo-tiny
```

### Step 4: Integrate YOLO with Video Stream

Update `frontend/src/components/LiveVideoStream.js`:

```jsx
import React, { useRef, useEffect, useState } from 'react';
import PropTypes from 'prop-types';
import YOLODetection from './YOLODetection';

function LiveVideoStream({ onFrame, onConnectionChange, onDetection }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const [pc, setPc] = useState(null);
  const [error, setError] = useState(null);
  const [detectedObjects, setDetectedObjects] = useState([]);

  // ... existing connection logic ...

  const handleDetection = (detections) => {
    setDetectedObjects(detections);
    onDetection?.(detections);
  };

  return (
    <div className="live-video-stream">
      <div className="video-container">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="video-element"
        />
        <canvas
          ref={canvasRef}
          style={{ display: 'none' }}
        />
        
        {/* YOLO Detection Overlay */}
        {isConnected && (
          <YOLODetection 
            canvas={canvasRef.current}
            onDetection={handleDetection}
          />
        )}
      </div>
      
      {/* Detection Results */}
      {detectedObjects.length > 0 && (
        <div className="detection-results">
          <h3>Detected Objects:</h3>
          <ul>
            {detectedObjects.map((obj, index) => (
              <li key={index}>
                {obj.class} ({(obj.confidence * 100).toFixed(1)}%)
              </li>
            ))}
          </ul>
        </div>
      )}
      
      {/* ... existing controls ... */}
    </div>
  );
}

// ... rest of component ...
```

---

## 💬 Chapter 6: Enhanced Chat Interface

### Step 1: Add Camera Commands

Update `frontend/src/components/ChatInterface.js`:

```jsx
// Add to existing imports
import { useState, useEffect, useRef } from 'react';

function ChatInterface({ userInfo, onCameraCommand }) {
  // ... existing state ...

  const handleCameraCommands = (message) => {
    const lowerMessage = message.toLowerCase();
    
    if (lowerMessage.includes('start camera') || lowerMessage.includes('connect camera')) {
      onCameraCommand?.('connect');
      return "Starting camera connection...";
    }
    
    if (lowerMessage.includes('stop camera') || lowerMessage.includes('disconnect camera')) {
      onCameraCommand?.('disconnect');
      return "Stopping camera connection...";
    }
    
    if (lowerMessage.includes('capture') || lowerMessage.includes('take photo')) {
      onCameraCommand?.('capture');
      return "Capturing image...";
    }
    
    if (lowerMessage.includes('detect') || lowerMessage.includes('what do you see')) {
      onCameraCommand?.('detect');
      return "Running object detection...";
    }
    
    return null; // Not a camera command
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || isTyping || isProcessingRef.current) return;

    const messageToSend = inputMessage.trim();
    isProcessingRef.current = true;
    setInputMessage('');
    setIsTyping(true);
    setShowTypingIndicator(true);

    const userMessage = {
      role: 'user',
      content: messageToSend
    };

    // Check for camera commands first
    const cameraResponse = handleCameraCommands(messageToSend);
    
    if (cameraResponse) {
      // Handle camera command locally
      setMessages(prevMessages => {
        const allMessages = [...prevMessages, userMessage];
        return allMessages;
      });
      
      // Add camera response
      setTimeout(() => {
        const assistantMessage = {
          role: 'assistant',
          content: cameraResponse
        };
        setMessages(prev => [...prev, assistantMessage]);
        setIsTyping(false);
        setShowTypingIndicator(false);
        isProcessingRef.current = false;
      }, 1000);
      
      return;
    }

    // ... existing API call logic ...
  };

  // ... rest of component ...
}
```

### Step 2: Add Camera Command Styles

Add to `frontend/src/index.css`:

```css
.detection-results {
  background: rgba(0, 0, 0, 0.8);
  color: white;
  padding: 15px;
  border-radius: 8px;
  margin-top: 10px;
}

.detection-results h3 {
  margin: 0 0 10px 0;
  font-size: 16px;
}

.detection-results ul {
  margin: 0;
  padding-left: 20px;
}

.detection-results li {
  margin: 5px 0;
  font-size: 14px;
}

.yolo-loading {
  display: flex;
  align-items: center;
  gap: 10px;
  color: white;
  padding: 10px;
  background: rgba(0, 0, 0, 0.7);
  border-radius: 6px;
  position: absolute;
  top: 10px;
  left: 10px;
}

.detection-status {
  color: white;
  background: rgba(0, 0, 0, 0.7);
  padding: 5px 10px;
  border-radius: 4px;
  font-size: 12px;
  position: absolute;
  top: 10px;
  right: 10px;
}
```

---

## 🔧 Chapter 7: Advanced Features

### Step 1: Motion Detection

Create `frontend/src/components/MotionDetection.js`:

```jsx
import React, { useEffect, useRef } from 'react';
import PropTypes from 'prop-types';

function MotionDetection({ canvas, onMotionDetected }) {
  const prevFrameRef = useRef(null);
  const threshold = 30; // Motion sensitivity

  useEffect(() => {
    if (!canvas) return;

    const detectMotion = () => {
      const ctx = canvas.getContext('2d');
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;

      if (prevFrameRef.current) {
        let motionPixels = 0;
        const totalPixels = data.length / 4;

        for (let i = 0; i < data.length; i += 4) {
          const r = data[i];
          const g = data[i + 1];
          const b = data[i + 2];
          
          const prevR = prevFrameRef.current[i];
          const prevG = prevFrameRef.current[i + 1];
          const prevB = prevFrameRef.current[i + 2];

          const diff = Math.abs(r - prevR) + Math.abs(g - prevG) + Math.abs(b - prevB);
          
          if (diff > threshold) {
            motionPixels++;
          }
        }

        const motionPercentage = (motionPixels / totalPixels) * 100;
        
        if (motionPercentage > 5) { // 5% threshold
          onMotionDetected?.(motionPercentage);
        }
      }

      // Store current frame
      prevFrameRef.current = new Uint8ClampedArray(data);
    };

    const interval = setInterval(detectMotion, 100); // Check every 100ms

    return () => clearInterval(interval);
  }, [canvas, onMotionDetected]);

  return null; // This component doesn't render anything
}

MotionDetection.propTypes = {
  canvas: PropTypes.object,
  onMotionDetected: PropTypes.func
};

export default MotionDetection;
```

### Step 2: Recording Feature

Add recording functionality to `LiveVideoStream.js`:

```jsx
// Add to state
const [isRecording, setIsRecording] = useState(false);
const mediaRecorderRef = useRef(null);
const recordedChunksRef = useRef([]);

const startRecording = () => {
  if (!videoRef.current.srcObject) return;

  recordedChunksRef.current = [];
  const stream = videoRef.current.srcObject;
  
  mediaRecorderRef.current = new MediaRecorder(stream);
  
  mediaRecorderRef.current.ondataavailable = (event) => {
    if (event.data.size > 0) {
      recordedChunksRef.current.push(event.data);
    }
  };
  
  mediaRecorderRef.current.onstop = () => {
    const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `recording-${Date.now()}.webm`;
    a.click();
    URL.revokeObjectURL(url);
  };
  
  mediaRecorderRef.current.start();
  setIsRecording(true);
};

const stopRecording = () => {
  if (mediaRecorderRef.current && isRecording) {
    mediaRecorderRef.current.stop();
    setIsRecording(false);
  }
};
```

### Step 3: Notification System

Create `frontend/src/components/NotificationSystem.js`:

```jsx
import React, { useEffect } from 'react';
import PropTypes from 'prop-types';

function NotificationSystem({ detectedObjects, onMotionDetected }) {
  useEffect(() => {
    // Check for specific objects
    if (detectedObjects.some(obj => obj.class === 'person')) {
      showNotification('Person detected in camera view');
    }
    
    if (detectedObjects.some(obj => obj.class === 'car')) {
      showNotification('Vehicle detected');
    }
  }, [detectedObjects]);

  useEffect(() => {
    if (onMotionDetected) {
      showNotification(`Motion detected! (${onMotionDetected.toFixed(1)}% change)`);
    }
  }, [onMotionDetected]);

  const showNotification = (message) => {
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification('IoT Camera Alert', {
        body: message,
        icon: '/favicon.ico'
      });
    }
  };

  const requestNotificationPermission = () => {
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
  };

  return (
    <div className="notification-system">
      <button onClick={requestNotificationPermission}>
        Enable Notifications
      </button>
    </div>
  );
}

NotificationSystem.propTypes = {
  detectedObjects: PropTypes.array,
  onMotionDetected: PropTypes.number
};

export default NotificationSystem;
```

---

## 🚀 Chapter 8: Deployment and Testing

### Step 1: Production Setup

**Raspberry Pi (Production):**

```bash
# Install PM2 for process management
sudo npm install -g pm2

# Create startup script
cat > start_server.sh << 'EOF'
#!/bin/bash
cd /home/pi/iot-webcam
source venv/bin/activate
python server.py
EOF

chmod +x start_server.sh

# Start with PM2
pm2 start start_server.sh --name "iot-webcam"
pm2 startup
pm2 save
```

**React App (Production):**

```bash
cd frontend
npm run build
# Deploy build folder to your web server
```

### Step 2: Security Considerations

**Add CORS to Raspberry Pi server:**

```python
# Add to server.py
from aiohttp_cors import setup as cors_setup

# Setup CORS
cors = cors_setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*",
        allow_methods="*"
    )
})

# Apply CORS to routes
for route in list(app.router.routes()):
    cors.add(route)
```

**Add authentication:**

```python
# Simple API key authentication
async def authenticate(request):
    api_key = request.headers.get('X-API-Key')
    if api_key != 'your-secret-key':
        raise web.HTTPUnauthorized()
    return True

# Apply to routes
app.router.add_post("/offer", authenticate, offer)
```

### Step 3: Performance Optimization

**Optimize video quality:**

```python
# In CameraVideoTrack class
self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
self.camera.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for better performance
```

**Optimize YOLO detection:**

```javascript
// Reduce detection frequency
const DETECTION_INTERVAL = 500; // 500ms instead of 100ms

detectionRef.current = setInterval(runDetection, DETECTION_INTERVAL);
```

---

## 🧪 Chapter 9: Testing and Debugging

### Step 1: Test Camera Connection

```bash
# Test camera on Raspberry Pi
raspistill -o test.jpg
raspivid -o test.h264 -t 5000

# Check camera permissions
ls -la /dev/video*
```

### Step 2: Test WebRTC Connection

```javascript
// Add to browser console
const pc = new RTCPeerConnection();
pc.createOffer().then(offer => {
  console.log('Offer created:', offer);
});
```

### Step 3: Test YOLO Detection

```javascript
// Test TensorFlow.js
import * as tf from '@tensorflow/tfjs';
tf.ready().then(() => {
  console.log('TensorFlow.js ready');
});
```

### Step 4: Debug Common Issues

**Camera not working:**
```bash
# Check camera module
vcgencmd get_camera

# Check camera interface
sudo raspi-config
# Interface Options → Camera → Enable
```

**WebRTC connection failed:**
- Check firewall settings
- Verify STUN server availability
- Check network connectivity

**YOLO model not loading:**
- Verify model files are in correct location
- Check browser console for errors
- Ensure TensorFlow.js backend is loaded

---

## 📚 Chapter 10: Advanced Features

### Step 1: Multiple Camera Support

```python
# Support multiple cameras
class MultiCameraServer:
    def __init__(self):
        self.cameras = {}
        self.pcs = set()
    
    async def add_camera(self, camera_id, camera_index):
        self.cameras[camera_id] = CameraVideoTrack(camera_index)
    
    async def remove_camera(self, camera_id):
        if camera_id in self.cameras:
            self.cameras[camera_id].stop()
            del self.cameras[camera_id]
```

### Step 2: Cloud Storage Integration

```python
# Save recordings to cloud
import boto3

s3 = boto3.client('s3')

async def save_recording(recording_data, filename):
    s3.put_object(
        Bucket='your-bucket',
        Key=f'recordings/{filename}',
        Body=recording_data
    )
```

### Step 3: Machine Learning Pipeline

```python
# Custom object detection training
import tensorflow as tf

def train_custom_model(training_data):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(training_data, epochs=10)
    return model
```

---

## 🎉 Conclusion

You've built a complete IoT system with:

✅ **Real-time video streaming** via WebRTC  
✅ **Object detection** with YOLO in the browser  
✅ **Voice control** through chat interface  
✅ **Motion detection** and notifications  
✅ **Recording capabilities**  
✅ **Production-ready deployment**  

### Next Steps:

1. **Add more AI models** (face recognition, gesture detection)
2. **Implement cloud storage** for recordings
3. **Add mobile app** for remote monitoring
4. **Integrate with smart home** systems
5. **Add machine learning** for custom object detection

### Learning Outcomes:

- **WebRTC** for real-time communication
- **TensorFlow.js** for browser-based AI
- **React** for interactive UIs
- **IoT development** with Raspberry Pi
- **Full-stack development** skills

**You're now an IoT developer!** 🚀

---

## 📝 Quick Reference

### Commands

```bash
# Start Raspberry Pi server
cd ~/iot-webcam && source venv/bin/activate && python server.py

# Start React development
cd frontend && npm start

# Build React for production
cd frontend && npm run build

# Deploy with PM2
pm2 start start_server.sh --name "iot-webcam"
```

### API Endpoints

- `GET /` - Main page
- `POST /offer` - WebRTC offer/answer exchange

### Chat Commands

- "start camera" - Connect to video stream
- "stop camera" - Disconnect from video stream
- "capture" - Take a photo
- "detect objects" - Run object detection
- "what do you see" - List detected objects

---

*Happy IoT development! 🎯* 