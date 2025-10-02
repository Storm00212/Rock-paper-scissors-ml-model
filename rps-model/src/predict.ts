// PREDICTION SCRIPT - Test the trained AI model
//
// This file loads a trained AI model and uses it to recognize hand gestures
// from images. It's like asking a student to identify objects in photos.
//
// WORKFLOW:
// 1. Load the trained AI model from disk
// 2. Find sample images from the dataset
// 3. Process each image (resize, normalize)
// 4. Ask AI to predict the gesture
// 5. Show results with confidence scores

// For Node.js execution (running in terminal)
import * as tf from "@tensorflow/tfjs"; // AI library
import * as fs from "fs"; // File operations
import * as path from "path"; // Path utilities
import sharp from "sharp"; // Image processing

// For browser compatibility (when running in web browser)
// declare const tf: any;

const CLASSES = ['rock', 'paper', 'scissors'];

let model: any;

function createModel() {
  const model = tf.sequential();

  // Same fast model as training: 16x16 images for speed
  model.add(tf.layers.conv2d({ inputShape: [16, 16, 3], filters: 8, kernelSize: 3, activation: "relu" }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 16, activation: "relu" }));
  model.add(tf.layers.dense({ units: CLASSES.length, activation: "softmax" }));

  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

async function loadModel() {
  try {
    model = await tf.loadLayersModel('./model/model.json');
    console.log('Fast model loaded from file (16x16, 6.5K params)');
  } catch (e) {
    console.log('Model file not found, creating fresh model for demo...');
    model = createModel();
    console.log('Fast model created (16x16, 6.5K params) - run "npm run train" to train it');
  }
}

async function predict(imagePath: string) {
  if (!model) await loadModel();

  // Fast processing: 16x16 images for speed (Node.js compatible)
  const imgBuffer = fs.readFileSync(imagePath);
  const resizedBuffer = await sharp(imgBuffer).resize(16, 16).raw().toBuffer();
  const tensor = tf.tensor3d(new Uint8Array(resizedBuffer), [16, 16, 3], 'float32')
    .div(255.0)
    .expandDims(0);

  const prediction = model.predict(tensor);
  const probabilities = prediction.dataSync();
  const predictedClass = CLASSES[probabilities.indexOf(Math.max(...probabilities))];

  return { predictedClass, probabilities: Array.from(probabilities) };
}

/**
 * TEST PREDICTION ON SAMPLE IMAGE
 *
 * This function finds a sample image from the dataset and tests the AI prediction.
 * It's like giving the AI a practice quiz to see how well it learned.
 *
 * WHAT IT DOES:
 * 1. Looks for dataset folders (rock/, paper/, scissors/)
 * 2. Finds the first PNG image in any folder
 * 3. Asks AI to predict the gesture
 * 4. Shows the result with confidence scores
 * 5. Displays percentages for each gesture type
 */
async function predictOnSampleImage() {
  // Define where to look for test images
  const datasetPath = path.join(__dirname, "../../dataset");
  const classes = ['rock', 'paper', 'scissors'];

  // Try each gesture folder
  for (const className of classes) {
    const classDir = path.join(datasetPath, className);
    if (fs.existsSync(classDir)) {
      // Get all PNG files in this folder
      const files = fs.readdirSync(classDir).filter(f => f.endsWith('.png'));
      if (files.length > 0) {
        // Use the first image found
        const sampleImage = path.join(classDir, files[0]!);
        console.log(`🖼️  Testing prediction on sample image: ${sampleImage}`);

        // Ask AI to predict
        const result = await predict(sampleImage);
        console.log(`🤖 Prediction: ${result.predictedClass}`);

        // Show confidence scores for all gestures
        console.log(`📊 Confidence: ${result.probabilities.map((p, i) =>
          `${classes[i]}: ${((p as number) * 100).toFixed(1)}%`
        ).join(', ')}`);

        return; // Stop after testing one image
      }
    }
  }

  console.log("❌ No sample images found in dataset. Please ensure dataset exists.");
}

/**
 * MAIN PREDICTION FUNCTION
 *
 * This is the entry point when you run `npm run predict`.
 * It sets up the AI environment and runs a prediction test.
 */
async function main() {
  // Initialize TensorFlow.js (same as training)
  await tf.setBackend('cpu');
  await tf.ready();

  console.log("🚀 Fast Rock-Paper-Scissors Prediction (16x16, 6.5K params)");
  console.log("💡 This script tests the AI on sample images from your dataset");

  // Load the trained AI model
  await loadModel();

  // Test prediction on a sample image
  await predictOnSampleImage();

  console.log("\n🎮 Want to play interactively? Run: npm run dev");
  console.log("🌐 Opens a beautiful web interface with camera gesture recognition!");
}

main().catch(console.error);
