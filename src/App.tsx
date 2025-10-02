// REACT FRONTEND - Interactive Rock Paper Scissors Game
//
// This file creates a beautiful web interface where users can play
// rock-paper-scissors against an AI using their camera for gesture recognition.
//
// FEATURES:
// - Real-time camera access and gesture detection
// - Beautiful animated UI with Tailwind CSS
// - Live game scoring and statistics
// - Auto-play mode every 3 seconds
// - Terminal synchronization for debugging
//
// HOW IT WORKS:
// 1. User grants camera permission
// 2. Camera feed displays on screen
// 3. AI analyzes hand gestures every 3 seconds
// 4. Game logic determines winner
// 5. Results show with animations and sounds
// 6. Scores update in real-time

// TensorFlow.js and MediaPipe are loaded from CDN (see index.html)
declare const tf: any;

// MediaPipe Hands types and functions
declare global {
  interface Window {
    Hands: any;
    Camera: any;
    drawConnectors: any;
    drawLandmarks: any;
    HAND_CONNECTIONS: any;
  }
}

import React, { useRef, useEffect, useState, useCallback } from 'react';

// GAME CONSTANTS
const CLASSES = ['rock', 'paper', 'scissors'] as const; // The 3 possible gestures
type Gesture = typeof CLASSES[number]; // TypeScript type for gestures

// DATA STRUCTURES
interface PredictionResult {
  gesture: Gesture;     // Which gesture was detected (rock/paper/scissors)
  confidence: number;   // How confident the AI is (0-1)
}

/**
 * MAIN APP COMPONENT
 *
 * This is the root React component that manages the entire game interface.
 * It handles camera access, AI predictions, game logic, and UI updates.
 */
const App: React.FC = () => {
  // REFS for DOM elements (direct access to HTML elements)
  const videoRef = useRef<HTMLVideoElement>(null);    // Camera video element
  const canvasRef = useRef<HTMLCanvasElement>(null);  // Hidden canvas for processing
  const overlayRef = useRef<HTMLCanvasElement>(null); // Canvas for skeleton overlay

  // STATE MANAGEMENT (React's way of storing changing data)
  const [model, setModel] = useState<any>(null);           // The trained AI model
  const [isModelLoading, setIsModelLoading] = useState(true); // Loading spinner state
  const [isCameraActive, setIsCameraActive] = useState(false); // Camera on/off
  const [currentPrediction, setCurrentPrediction] = useState<PredictionResult | null>(null); // Latest AI prediction
  const [computerChoice, setComputerChoice] = useState<Gesture | null>(null); // Computer's gesture
  const [gameResult, setGameResult] = useState<'win' | 'lose' | 'tie' | null>(null); // Who won?
  const [score, setScore] = useState({ player: 0, computer: 0, ties: 0 }); // Game statistics
  const [isPlaying, setIsPlaying] = useState(false);      // Prevents overlapping games

  // HAND DETECTION STATE
  const [hands, setHands] = useState<any>(null);          // MediaPipe Hands instance
  const [handDetected, setHandDetected] = useState(false); // Is a hand currently detected?
  const [handBoundingBox, setHandBoundingBox] = useState<{x: number, y: number, width: number, height: number} | null>(null); // Hand location
  const [handLandmarks, setHandLandmarks] = useState<any[]>([]); // Current hand keypoints for skeleton drawing

  // Load the fast model and initialize hand detection
  useEffect(() => {
    const loadModel = async () => {
      try {
        const loadedModel = await tf.loadLayersModel('./model/model.json');
        setModel(loadedModel);
        console.log('🚀 Fast AI model loaded (16x16, 6.5K params)');
      } catch (error) {
        console.log('Model not found, creating demo model...');
        // Create the same fast model architecture for demo
        const demoModel = tf.sequential();
        demoModel.add(tf.layers.conv2d({ inputShape: [16, 16, 3], filters: 8, kernelSize: 3, activation: "relu" }));
        demoModel.add(tf.layers.maxPooling2d({ poolSize: 2 }));
        demoModel.add(tf.layers.flatten());
        demoModel.add(tf.layers.dense({ units: 16, activation: "relu" }));
        demoModel.add(tf.layers.dense({ units: CLASSES.length, activation: "softmax" }));
        demoModel.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
        setModel(demoModel);
        console.log('🎮 Demo model created - run training for better accuracy');
      }
      setIsModelLoading(false);
    };

    const initializeHandDetection = async () => {
      // Initialize MediaPipe Hands for hand detection
      const handsInstance = new window.Hands({
        locateFile: (file: string) => {
          return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`;
        }
      });

      handsInstance.setOptions({
        maxNumHands: 1,           // Detect only one hand
        modelComplexity: 0,       // Fastest model (0-2, higher = more accurate but slower)
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });

      // Set up hand detection callback
      handsInstance.onResults((results: any) => {
        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
          setHandDetected(true);
          setHandLandmarks(results.multiHandLandmarks[0]); // Store landmarks for skeleton drawing

          // Calculate hand bounding box from landmarks
          const landmarks = results.multiHandLandmarks[0];
          let minX = 1, minY = 1, maxX = 0, maxY = 0;

          landmarks.forEach((landmark: any) => {
            minX = Math.min(minX, landmark.x);
            minY = Math.min(minY, landmark.y);
            maxX = Math.max(maxX, landmark.x);
            maxY = Math.max(maxY, landmark.y);
          });

          // Add some padding around the hand
          const padding = 0.1;
          const width = maxX - minX;
          const height = maxY - minY;

          setHandBoundingBox({
            x: Math.max(0, minX - padding * width),
            y: Math.max(0, minY - padding * height),
            width: Math.min(1, width + 2 * padding * width),
            height: Math.min(1, height + 2 * padding * height)
          });

          // Draw the skeleton overlay
          drawHandSkeleton(landmarks);
        } else {
          setHandDetected(false);
          setHandBoundingBox(null);
          setHandLandmarks([]);

          // Clear the skeleton overlay
          const overlay = overlayRef.current;
          if (overlay) {
            const ctx = overlay.getContext('2d');
            if (ctx) {
              ctx.clearRect(0, 0, overlay.width, overlay.height);
            }
          }
        }
      });

      setHands(handsInstance);
      console.log('🤖 Hand detection initialized with MediaPipe');
    };

    loadModel();
    initializeHandDetection();
  }, []);

  // Initialize camera
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 320, height: 240, facingMode: 'user' }
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraActive(true);
      }
    } catch (error) {
      console.error('Camera access denied:', error);
      alert('Camera access is required for gesture recognition!');
    }
  }, []);

  // Stop camera
  const stopCamera = useCallback(() => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      setIsCameraActive(false);
    }
  }, []);

  // Draw hand skeleton overlay
  const drawHandSkeleton = useCallback((landmarks: any[]) => {
    const overlay = overlayRef.current;
    if (!overlay) return;

    const ctx = overlay.getContext('2d');
    if (!ctx) return;

    // Clear previous drawing
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Set canvas size to match video
    const video = videoRef.current;
    if (video) {
      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;
    }

    if (!landmarks || landmarks.length === 0) return;

    const videoWidth = overlay.width;
    const videoHeight = overlay.height;

    // MediaPipe hand connections (which keypoints connect to which)
    const HAND_CONNECTIONS = [
      // Thumb
      [0, 1], [1, 2], [2, 3], [3, 4],
      // Index finger
      [0, 5], [5, 6], [6, 7], [7, 8],
      // Middle finger
      [0, 9], [9, 10], [10, 11], [11, 12],
      // Ring finger
      [0, 13], [13, 14], [14, 15], [15, 16],
      // Pinky
      [0, 17], [17, 18], [18, 19], [19, 20],
      // Palm connections
      [5, 9], [9, 13], [13, 17]
    ];

    // Draw connections (lines)
    ctx.strokeStyle = '#00ff00'; // Bright green
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';

    HAND_CONNECTIONS.forEach(([start, end]) => {
      const startPoint = landmarks[start];
      const endPoint = landmarks[end];

      if (startPoint && endPoint) {
        ctx.beginPath();
        ctx.moveTo(startPoint.x * videoWidth, startPoint.y * videoHeight);
        ctx.lineTo(endPoint.x * videoWidth, endPoint.y * videoHeight);
        ctx.stroke();
      }
    });

    // Draw keypoints (dots)
    ctx.fillStyle = '#ff0000'; // Bright red
    landmarks.forEach((landmark: any) => {
      const x = landmark.x * videoWidth;
      const y = landmark.y * videoHeight;

      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI);
      ctx.fill();

      // Add white border for better visibility
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1;
      ctx.stroke();
    });
  }, []);

  /**
   * PREDICT GESTURE FROM CAMERA WITH HAND DETECTION
   *
   * This function captures a frame from the camera, detects hands, crops to focus
   * on the hand, and asks the AI to identify the gesture. Much more accurate!
   *
   * STEPS:
   * 1. Capture current camera frame
   * 2. Use MediaPipe to detect hand location
   * 3. Crop image to focus only on the detected hand
   * 4. Resize to 16x16 (matches training size)
   * 5. Convert to AI-friendly format
   * 6. Get prediction from model
   * 7. Return gesture + confidence score
   */
  const predictGesture = useCallback(async (): Promise<PredictionResult | null> => {
    // Check if everything is ready
    if (!model || !videoRef.current || !canvasRef.current) return null;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    // STEP 1: Capture full camera frame first
    const video = videoRef.current;
    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;

    // Set canvas to video dimensions for full frame capture
    canvas.width = videoWidth;
    canvas.height = videoHeight;
    ctx.drawImage(video, 0, 0, videoWidth, videoHeight);

    // STEP 2: Send frame to MediaPipe for hand detection
    if (hands) {
      await hands.send({ image: canvas });
    }

    let processedImageData: ImageData;

    // STEP 3: Crop to hand if detected, otherwise use center region
    if (handDetected && handBoundingBox) {
      // Crop to detected hand bounding box
      const box = handBoundingBox;
      const cropX = box.x * videoWidth;
      const cropY = box.y * videoHeight;
      const cropWidth = box.width * videoWidth;
      const cropHeight = box.height * videoHeight;

      // Extract hand region
      processedImageData = ctx.getImageData(cropX, cropY, cropWidth, cropHeight);
    } else {
      // No hand detected - use center region as fallback
      const centerSize = Math.min(videoWidth, videoHeight) * 0.6; // 60% of smaller dimension
      const centerX = (videoWidth - centerSize) / 2;
      const centerY = (videoHeight - centerSize) / 2;

      processedImageData = ctx.getImageData(centerX, centerY, centerSize, centerSize);
    }

    // STEP 4: Create a temporary canvas for resizing
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 16;
    tempCanvas.height = 16;
    const tempCtx = tempCanvas.getContext('2d')!;

    // Create image from cropped data and resize
    const tempImg = new ImageData(processedImageData.data, processedImageData.width, processedImageData.height);
    const tempCanvas2 = document.createElement('canvas');
    tempCanvas2.width = processedImageData.width;
    tempCanvas2.height = processedImageData.height;
    tempCanvas2.getContext('2d')!.putImageData(tempImg, 0, 0);

    // Resize to 16x16
    tempCtx.drawImage(tempCanvas2, 0, 0, 16, 16);

    // STEP 5: Convert to AI tensor format
    const finalImageData = tempCtx.getImageData(0, 0, 16, 16);
    const tensor = tf.browser.fromPixels(finalImageData)  // Convert pixels to tensor
      .toFloat()                                          // Convert to floating point
      .div(255.0)                                         // Normalize: 0-255 → 0-1
      .expandDims(0);                                     // Add batch dimension [1,16,16,3]

    // STEP 6: Ask AI to predict
    const prediction = model.predict(tensor) as any;
    const probabilities = prediction.dataSync();          // Get prediction scores

    // STEP 7: Find the gesture with highest confidence
    const maxIndex = probabilities.indexOf(Math.max(...probabilities));
    const gesture = CLASSES[maxIndex] as Gesture;
    const confidence = probabilities[maxIndex];

    // STEP 8: Clean up memory (prevents memory leaks!)
    tensor.dispose();
    prediction.dispose();

    return { gesture, confidence };
  }, [model, hands, handDetected, handBoundingBox]);

  /**
   * DETERMINE GAME WINNER
   *
   * This function implements the classic rock-paper-scissors rules:
   * - Rock beats Scissors
   * - Paper beats Rock
   * - Scissors beats Paper
   * - Same gestures = tie
   */
  const determineWinner = (player: Gesture, computer: Gesture): 'win' | 'lose' | 'tie' => {
    if (player === computer) return 'tie'; // Same gesture = tie

    // Rock-paper-scissors rules: what each gesture beats
    const wins = {
      rock: 'scissors',     // Rock beats scissors
      paper: 'rock',        // Paper beats rock
      scissors: 'paper'     // Scissors beats paper
    };

    // If player's gesture beats computer's gesture = win, otherwise = lose
    return wins[player] === computer ? 'win' : 'lose';
  };

  /**
   * PLAY A ROUND OF THE GAME
   *
   * This function orchestrates one complete round of rock-paper-scissors:
   * 1. Capture player's gesture from camera
   * 2. Computer makes random choice
   * 3. Determine who wins
   * 4. Update scores and UI
   * 5. Log results to terminal
   *
   * Called automatically every 3 seconds when camera is active!
   */
  const playRound = useCallback(async () => {
    if (!isCameraActive || isPlaying) return; // Prevent overlapping rounds

    setIsPlaying(true);     // Block new rounds until this one finishes
    setGameResult(null);    // Clear previous result

    try {
      // STEP 1: Get player's gesture from camera
      const playerResult = await predictGesture();
      if (!playerResult) {
        setIsPlaying(false);
        return;
      }

      setCurrentPrediction(playerResult); // Show on UI

      // STEP 2: Computer makes random choice (fair play!)
      const computerGesture = CLASSES[Math.floor(Math.random() * CLASSES.length)] as Gesture;
      setComputerChoice(computerGesture);

      // STEP 3: Apply rock-paper-scissors rules
      const result = determineWinner(playerResult.gesture, computerGesture);
      setGameResult(result);

      // STEP 4: Update score statistics
      setScore(prev => ({
        ...prev,
        // Increment the appropriate counter based on result
        [result === 'win' ? 'player' : result === 'lose' ? 'computer' : 'ties']:
          prev[result === 'win' ? 'player' : result === 'lose' ? 'computer' : 'ties'] + 1
      }));

      // STEP 5: Log to terminal (as requested by user)
      console.log(`🎮 Round Result: Player (${playerResult.gesture}) vs Computer (${computerGesture}) = ${result.toUpperCase()}`);
      console.log(`📊 Score: Player ${score.player + (result === 'win' ? 1 : 0)} - Computer ${score.computer + (result === 'lose' ? 1 : 0)} - Ties ${score.ties + (result === 'tie' ? 1 : 0)}`);

    } catch (error) {
      console.error('Prediction error:', error);
    } finally {
      setIsPlaying(false); // Allow next round to start
    }
  }, [isCameraActive, isPlaying, predictGesture, score]);

  // Auto-play every 3 seconds when camera is active
  useEffect(() => {
    if (!isCameraActive) return;

    const interval = setInterval(() => {
      if (!isPlaying) {
        playRound();
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [isCameraActive, isPlaying, playRound]);

  // Cleanup on unmount
  useEffect(() => {
    return () => stopCamera();
  }, [stopCamera]);

  const getGestureEmoji = (gesture: Gesture) => {
    const emojis = { rock: '🪨', paper: '📄', scissors: '✂️' };
    return emojis[gesture];
  };

  const getResultMessage = (result: 'win' | 'lose' | 'tie') => {
    const messages = {
      win: '🎉 You Win!',
      lose: '😢 Computer Wins!',
      tie: '🤝 It\'s a Tie!'
    };
    return messages[result];
  };

  if (isModelLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-primary-50 to-secondary-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin-slow text-6xl mb-4">🤖</div>
          <h1 className="text-2xl font-bold text-gray-800 mb-2">Loading AI Model...</h1>
          <p className="text-gray-600">Fast 16×16 gesture recognition initializing</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 to-secondary-50 p-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            🤖 Rock Paper Scissors AI
          </h1>
          <p className="text-lg text-gray-600">
            Fast gesture recognition with TensorFlow.js
          </p>
          <div className="mt-4 text-sm text-gray-500">
            ⚡ 16×16 processing • 6.5K parameters • Real-time prediction
          </div>
        </div>

        {/* Score Board */}
        <div className="card mb-6">
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-primary-600">{score.player}</div>
              <div className="text-sm text-gray-600">You</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-600">{score.ties}</div>
              <div className="text-sm text-gray-600">Ties</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-red-600">{score.computer}</div>
              <div className="text-sm text-gray-600">Computer</div>
            </div>
          </div>
        </div>

        {/* Camera and Game Area */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* Camera Feed */}
          <div className="card">
            <h2 className="text-xl font-bold mb-4 text-center">📹 Camera Feed</h2>
            <div className="relative">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full rounded-lg border-2 border-gray-200"
                style={{ display: isCameraActive ? 'block' : 'none' }}
              />
              {/* Skeleton overlay canvas */}
              <canvas
                ref={overlayRef}
                className="absolute top-0 left-0 w-full h-full pointer-events-none rounded-lg"
                style={{
                  display: isCameraActive ? 'block' : 'none',
                  mixBlendMode: 'multiply' // Makes overlay blend nicely
                }}
              />
              <canvas
                ref={canvasRef}
                width={16}
                height={16}
                className="hidden"
              />
              {!isCameraActive && (
                <div className="aspect-video bg-gray-100 rounded-lg flex items-center justify-center">
                  <div className="text-center">
                    <div className="text-4xl mb-2">📷</div>
                    <p className="text-gray-600">Camera not active</p>
                  </div>
                </div>
              )}
            </div>
            <div className="mt-4 flex gap-2">
              {!isCameraActive ? (
                <button onClick={startCamera} className="btn-primary flex-1">
                  🎥 Start Camera
                </button>
              ) : (
                <button onClick={stopCamera} className="btn-secondary flex-1">
                  ⏹️ Stop Camera
                </button>
              )}
            </div>
          </div>

          {/* Game Results */}
          <div className="card">
            <h2 className="text-xl font-bold mb-4 text-center">🎮 Game Results</h2>

            {/* Current Round */}
            {currentPrediction && (
              <div className="mb-4 p-4 bg-gray-50 rounded-lg">
                <div className="flex justify-between items-center mb-2">
                  <div className="text-center">
                    <div className="gesture-icon">{getGestureEmoji(currentPrediction.gesture)}</div>
                    <div className="text-sm font-medium">You</div>
                    <div className="text-xs text-gray-600">
                      {(currentPrediction.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="text-2xl font-bold text-gray-400">VS</div>
                  <div className="text-center">
                    <div className="gesture-icon">
                      {computerChoice ? getGestureEmoji(computerChoice) : '🤔'}
                    </div>
                    <div className="text-sm font-medium">Computer</div>
                  </div>
                </div>
              </div>
            )}

            {/* Result Announcement */}
            {gameResult && (
              <div className={`result-announcement ${gameResult}`}>
                {getResultMessage(gameResult)}
              </div>
            )}

            {/* Status */}
            <div className="mt-4 text-center space-y-2">
              {isPlaying ? (
                <div className="flex items-center justify-center gap-2">
                  <div className="animate-spin-slow">🔄</div>
                  <span className="text-lg font-medium">Analyzing gesture...</span>
                </div>
              ) : isCameraActive ? (
                <div className="space-y-1">
                  <div className="text-green-600 font-medium">
                    ✅ Camera active - Auto-playing every 3 seconds
                  </div>
                  <div className={`text-sm font-medium ${handDetected ? 'text-blue-600' : 'text-orange-600'}`}>
                    {handDetected ? '🤖 Hand detected - AI focused on gesture' : '👋 Position hand in frame for better accuracy'}
                  </div>
                </div>
              ) : (
                <div className="text-gray-500">
                  📷 Start camera to begin playing
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Instructions */}
        <div className="card mt-6">
          <h3 className="text-lg font-bold mb-2">🎯 How to Play</h3>
          <ul className="text-sm text-gray-600 space-y-1">
            <li>• Click "Start Camera" to enable gesture recognition</li>
            <li>• Position your hand clearly in the camera frame</li>
            <li>• Watch for the red skeleton dots and green connecting lines on your hand</li>
            <li>• Make rock ✊, paper ✋, or scissors ✌️ gestures</li>
            <li>• AI automatically detects and crops to your hand for accuracy</li>
            <li>• Watch for "🤖 Hand detected" message for best results</li>
            <li>• Results appear instantly with confidence scores</li>
            <li>• Check the browser console for detailed logs</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default App;