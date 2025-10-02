# 🤖 Rock Paper Scissors AI - Complete Gesture Recognition Game

**A complete AI-powered rock-paper-scissors game with real-time camera gesture recognition!**

## 🎯 What This Project Does

This project creates an AI that can recognize hand gestures (rock ✊, paper ✋, scissors ✌️) using your computer's camera. You play rock-paper-scissors against the computer, and the AI detects your gestures automatically!

### ✨ Key Features
- **Ultra-Fast AI Training** (< 1 second)
- **Advanced Hand Detection** with MediaPipe tracking
- **Real-Time Skeleton Overlay** (21 keypoints + connecting lines)
- **Smart Gesture Recognition** (crops to focus on hands)
- **Real-Time Camera Processing** with auto-play
- **Beautiful Web Interface** with animations
- **Live Scoring** and statistics
- **Terminal + Browser** synchronized output

## 🚀 Quick Start (For Beginners)

### Step 1: Install Dependencies
```bash
npm install
```

### Step 2: Train the AI Model
```bash
npm run train
```
This trains the AI in under 1 second using optimized algorithms!

### Step 3: Test AI Predictions
```bash
npm run predict
```
Tests the AI on sample images from your dataset.

### Step 4: Play the Game!
```bash
npm run dev
```
Opens a beautiful web interface at `http://localhost:1234`

## 📁 Project Structure (What Each File Does)

```
rps-ai/
├── rps-model/                    # AI Model Code
│   ├── src/
│   │   ├── train.ts             # Trains the AI model
│   │   └── predict.ts           # Tests AI predictions
│   └── ...
├── src/                         # React Web App
│   ├── App.tsx                  # Main game interface
│   ├── index.tsx                # App entry point
│   ├── index.html               # HTML template
│   └── index.css                # Styles (Tailwind CSS)
├── dataset/                     # Training images (you provide these)
│   ├── rock/                    # Rock gesture photos
│   ├── paper/                   # Paper gesture photos
│   └── scissors/                # Scissors gesture photos
└── package.json                 # Project configuration
```

## 🧠 How the AI Works (Beginner-Friendly Explanation)

### What is AI Training?
1. **Show Examples**: We give the AI thousands of photos of rock, paper, and scissors gestures
2. **Find Patterns**: AI learns what makes a "rock" look different from "paper"
3. **Make Predictions**: When it sees a new gesture, it guesses which one it is

### Our AI Architecture
```
Input Image (16×16 pixels)
    ↓
Convolutional Layer (finds shapes/edges)
    ↓
Pooling Layer (makes image smaller)
    ↓
Flatten (2D → 1D)
    ↓
Dense Layers (makes final decision)
    ↓
Output: Rock/Paper/Scissors probabilities
```

### Why It's Fast
- **Small Images**: 16×16 instead of 64×64 (16x less data)
- **Simple Model**: 6,563 parameters instead of 400K+
- **Smart Data Loading**: Loads images in batches, not all at once
- **Optimized Training**: Only 3 epochs, higher learning rate

## 🎮 How to Play

1. **Open the Game**: Run `npm run dev` and go to `http://localhost:1234`
2. **Allow Camera**: Click "Start Camera" and grant permission
3. **Position Hand**: Make sure your hand is clearly visible in the frame
4. **Watch Detection**: Look for "🤖 Hand detected" message for best accuracy
5. **Make Gestures**: Hold up rock ✊, paper ✋, or scissors ✌️
6. **Watch AI Play**: The computer responds automatically every 3 seconds
7. **See Results**: Win/lose/tie announcements with animations
8. **Check Scores**: Live statistics update in real-time

## 🤖 Advanced Hand Detection

The AI now uses **MediaPipe Hands** for professional-grade hand tracking:

### How It Works
1. **Hand Detection**: MediaPipe analyzes the camera feed to locate hands
2. **Landmark Tracking**: Identifies 21 key points on each hand
3. **Bounding Box**: Calculates the exact hand region
4. **Smart Cropping**: AI focuses only on the detected hand (not background)
5. **Gesture Classification**: Processes the cropped hand for rock/paper/scissors

### Accuracy Improvements
- **Before**: AI looked at entire camera frame (confusing backgrounds)
- **After**: AI crops to focus only on your hand (much more accurate!)
- **Fallback**: If no hand detected, uses center region as backup

### Visual Feedback
- **Blue Message**: "🤖 Hand detected - AI focused on gesture" (optimal)
- **Orange Message**: "👋 Position hand in frame for better accuracy" (fallback)

## 🤖 Real-Time Skeleton Overlay

The AI now shows exactly what it's "seeing" with a live skeleton overlay!

### What You'll See
- **🔴 Red Dots**: 21 hand keypoints (joints and fingertips)
- **🟢 Green Lines**: Connecting lines forming the hand skeleton
- **Real-Time Tracking**: Skeleton moves as you move your hand
- **Professional Visualization**: Same tech used in AR filters and motion capture

### How It Works
1. **MediaPipe Detects**: AI finds your hand in the camera feed
2. **21 Keypoints**: Identifies exact positions of all hand joints
3. **Skeleton Drawing**: Connects keypoints with anatomical lines
4. **Live Overlay**: Updates 30+ times per second for smooth tracking

### Why It's Amazing
- **Visual Debugging**: See exactly what the AI is tracking
- **Accuracy Feedback**: Know when hand detection is working perfectly
- **Educational**: Learn how pose estimation works in real-time
- **Professional Look**: Same visualization as research papers and demos

### Skeleton Anatomy
```
Fingertips → Knuckles → Palm → Wrist
    ↓         ↓        ↓      ↓
   🔴        🔴       🔴     🔴
    🟢━━━━━━━━🟢━━━━━━━🟢━━━━━🟢
```

## 🔧 Technical Details

### Technologies Used
- **TensorFlow.js**: AI library for JavaScript
- **React**: Web user interface framework
- **Tailwind CSS**: Beautiful styling system
- **Parcel**: Fast web app bundler
- **Sharp**: Image processing library

### Performance Optimizations
- **Memory Efficient**: Streaming data pipeline prevents memory crashes
- **Fast Inference**: 16×16 processing for real-time recognition
- **WebGL Backend**: Hardware-accelerated AI computations
- **Batch Processing**: Trains on multiple images simultaneously

### Data Pipeline
```
Raw Images → Resize → Normalize → AI Model → Predictions
     ↓           ↓         ↓          ↓          ↓
  200×200    16×16    0-1 values  Neural Net  Rock/Paper/Scissors
```

## 🗂️ Setting Up Your Dataset

Create a `dataset/` folder with subfolders:
```
dataset/
├── rock/       # Photos of rock gestures
├── paper/      # Photos of paper gestures
└── scissors/   # Photos of scissors gestures
```

Each folder should contain PNG images of hand gestures from different angles and lighting conditions.

## 🎯 Learning Outcomes

By exploring this project, you'll learn:
- **Machine Learning**: How AI learns from data
- **Computer Vision**: How computers "see" and understand images
- **React Development**: Building interactive web applications
- **Real-Time Systems**: Handling camera input and live processing
- **Performance Optimization**: Making AI fast enough for real-time use

## 🚨 Troubleshooting

### "Camera not accessible"
- Make sure you're using a modern web browser (Chrome/Firefox/Edge)
- Grant camera permissions when prompted
- Try refreshing the page

### "Model not found"
- Run `npm run train` first to create the AI model
- Check that the training completed successfully

### "Out of memory" errors
- The optimized data pipeline prevents most memory issues
- If you have very large datasets, the subset selection keeps it manageable

## 🎉 What Makes This Special

1. **Educational**: Extensive comments explain every concept
2. **Optimized**: Runs fast even on modest hardware
3. **Complete**: Training + prediction + beautiful UI
4. **Real-World**: Actual camera integration, not just theory
5. **Scalable**: Can handle more data without performance loss

## 🔄 Workflow Summary

```
1. Collect gesture photos → 2. Train AI model → 3. Test predictions → 4. Build web game → 5. Play with camera!
   (Your dataset)           (npm run train)     (npm run predict)     (npm run dev)      (Browser game)
```

**Enjoy your AI-powered rock-paper-scissors game! 🎮🤖✨**
