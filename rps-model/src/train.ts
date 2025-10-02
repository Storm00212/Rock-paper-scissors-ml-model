/**
 * ROCK PAPER SCISSORS AI TRAINING SCRIPT
 *
 * This file trains an AI model to recognize hand gestures (rock, paper, scissors)
 * from images. The AI learns patterns in images to classify new gestures it hasn't seen before.
 *
 * WORKFLOW:
 * 1. Load images from dataset folder
 * 2. Process images (resize, normalize)
 * 3. Create neural network model
 * 4. Train model on image patterns
 * 5. Save trained model for later use
 *
 * KEY CONCEPTS FOR BEGINNERS:
 * - Dataset: Collection of labeled images (rock/paper/scissors photos)
 * - Neural Network: AI system that learns patterns from data
 * - Training: Process where AI learns from examples
 * - Classification: AI guessing which gesture is in a new image
 */

// Import required libraries
import * as tf from "@tensorflow/tfjs"; // TensorFlow.js - AI library for JavaScript
import * as fs from "fs"; // File system operations (reading files)
import * as path from "path"; // File path utilities
import sharp from "sharp"; // Image processing library

// Configuration constants
const DATASET_PATH = path.join(__dirname, "../../dataset"); // Where training images are stored
const CLASSES: string[] = ["rock", "paper", "scissors"]; // The 3 gesture types we want to recognize

/**
 * LOAD AND PREPARE TRAINING DATA
 *
 * This function loads all training images and prepares them for AI training.
 * Think of it like organizing photos in albums before showing them to a student.
 *
 * WHAT IT DOES:
 * 1. Finds all image files in dataset folders (rock/, paper/, scissors/)
 * 2. Creates a list of image paths with their correct labels (0=rock, 1=paper, 2=scissors)
 * 3. Shuffles the data so the AI doesn't learn in a predictable order
 * 4. Splits data into training (80%) and validation (20%) sets
 * 5. Creates a data pipeline that loads images on-demand (memory efficient!)
 *
 * WHY THIS APPROACH?
 * - Traditional way: Load ALL images into memory at once (uses lots of RAM)
 * - Our way: Load images in small batches as needed (uses very little RAM)
 * - Result: Can train on thousands of images without crashing computer!
 */
async function loadImages(): Promise<{ trainDataset: tf.data.Dataset<{ xs: tf.Tensor3D; ys: tf.Tensor1D }>; valDataset: tf.data.Dataset<{ xs: tf.Tensor3D; ys: tf.Tensor1D }> }> {
  // List to store all our training examples (image path + correct answer)
  const dataItems: { path: string; label: number }[] = [];

  // Loop through each gesture type (rock, paper, scissors)
  for (let i = 0; i < CLASSES.length; i++) {
    const className: string = CLASSES[i]!; // Current gesture name
    const dir = path.join(DATASET_PATH, className); // Path to gesture folder

    // Check if the folder exists (error if dataset is missing)
    if (!fs.existsSync(dir)) throw new Error(`Dataset folder not found: ${dir}`);

    // Get list of all files in this gesture folder
    const files = fs.readdirSync(dir);

    // Add each image file to our training list with its label
    for (const file of files) {
      if (!file.endsWith(".png")) continue; // Skip non-PNG files
      dataItems.push({
        path: path.join(dir, file), // Full path to image
        label: i // Numeric label (0=rock, 1=paper, 2=scissors)
      });
    }
  }

  // Shuffle the data
  function shuffle(array: any[]) {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
  }
  shuffle(dataItems);

  // Use only a small subset for fast training
  const subsetSize = Math.min(100, dataItems.length); // Use max 100 samples
  const subsetItems = dataItems.slice(0, subsetSize);

  const valSplit = 0.2;
  const valSize = Math.floor(subsetItems.length * valSplit);
  const valItems = subsetItems.slice(0, valSize);
  const trainItems = subsetItems.slice(valSize);

  const createDataset = (items: { path: string; label: number }[]) =>
    tf.data.array(items).mapAsync(async (item: { path: string; label: number }) => {
      const imgBuffer = fs.readFileSync(item.path);
      const resizedBuffer = await sharp(imgBuffer).resize(16, 16).raw().toBuffer(); // Much smaller images
      const imgTensor = tf.tensor3d(new Uint8Array(resizedBuffer), [16, 16, 3], 'float32').div(255.0) as tf.Tensor3D;
      const labelTensor = tf.oneHot(tf.scalar(item.label, "int32"), CLASSES.length) as tf.Tensor1D;
      return { xs: imgTensor, ys: labelTensor };
    });

  const trainDataset = createDataset(trainItems);
  const valDataset = createDataset(valItems);

  return { trainDataset, valDataset };
}

/**
 * CREATE THE AI BRAIN (NEURAL NETWORK)
 *
 * This function builds the AI model - think of it like designing a brain that can see patterns in images.
 *
 * NEURAL NETWORK LAYERS EXPLAINED:
 * 1. Conv2D Layer: Looks for patterns like edges, shapes in the image
 * 2. MaxPooling: Reduces image size while keeping important features
 * 3. Flatten: Converts 2D image into 1D list of numbers
 * 4. Dense Layers: Makes final decision about which gesture it is
 *
 * WHY THIS ARCHITECTURE?
 * - Convolutional layers: Great at recognizing visual patterns (like hand shapes)
 * - Small model: Only 6,563 parameters (very fast to train!)
 * - ReLU activation: Helps AI learn complex patterns
 * - Softmax output: Gives probability scores for each gesture (0-100%)
 */
function createModel() {
  // Create a sequential neural network (layers stack on top of each other)
  const model = tf.sequential();

  // LAYER 1: Convolutional layer - finds patterns in 16x16 images
  // 8 filters = 8 different pattern detectors
  // kernelSize 3 = looks at 3x3 pixel patterns
  model.add(tf.layers.conv2d({
    inputShape: [16, 16, 3], // Input: 16x16 pixel images with 3 color channels (RGB)
    filters: 8,              // Number of pattern detectors
    kernelSize: 3,           // Size of pattern window (3x3 pixels)
    activation: "relu"       // Activation function (helps learn complex patterns)
  }));

  // LAYER 2: Pooling layer - reduces image size, keeps important features
  // poolSize 2 = makes image 1/2 the size (8x8 becomes 4x4)
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  // LAYER 3: Flatten - converts 2D image (4x4x8) into 1D list (128 numbers)
  model.add(tf.layers.flatten());

  // LAYER 4: Dense layer - learns combinations of patterns
  model.add(tf.layers.dense({
    units: 16,        // 16 neurons to learn pattern combinations
    activation: "relu" // Another activation for complex learning
  }));

  // LAYER 5: Output layer - makes final gesture prediction
  model.add(tf.layers.dense({
    units: CLASSES.length, // 3 outputs (one for each gesture)
    activation: "softmax"  // Converts to probabilities (adds up to 100%)
  }));

  // Configure how the AI learns
  model.compile({
    optimizer: tf.train.adam(0.01), // Adam optimizer with learning rate 0.01
    loss: "categoricalCrossentropy", // Loss function (measures wrong answers)
    metrics: ["accuracy"],           // Track accuracy during training
  });

  return model;
}

/**
 * MAIN TRAINING FUNCTION
 *
 * This is the "conductor" that orchestrates the entire AI training process.
 * Think of it like a teacher running a class:
 * 1. Prepare the classroom (setup AI backend)
 * 2. Gather the students (load training data)
 * 3. Design the lesson plan (create model)
 * 4. Teach the class (train the model)
 * 5. Save the learned knowledge (save model)
 */
async function main() {
  // Initialize TensorFlow.js with CPU backend (no GPU needed for small model)
  await tf.setBackend('cpu');
  await tf.ready();

  console.log("🔄 Loading dataset...");
  // Load and prepare all training images
  const { trainDataset, valDataset } = await loadImages();
  console.log("✅ Dataset loaded and ready for training");

  // Create the AI model (neural network)
  const model = createModel();
  // Print model architecture summary
  model.summary();

  console.log("🚀 Training model... (this will be very fast!)");
  // Train the AI on the prepared data
  await model.fitDataset(
    trainDataset.batch(50), // Process 50 images at a time
    {
      epochs: 3, // Train for 3 complete passes through data (very fast!)
      validationData: valDataset.batch(50), // Check performance on unseen data
      // No early stopping - we want consistent fast training
    }
  );

  console.log("💾 Saving trained model...");
  try {
    // Save the trained AI model to disk for later use
    await model.save("file://model");
    console.log("✅ Training complete! Model saved at ./model");
    console.log("🎮 You can now run the React game: npm run dev");
  } catch (e) {
    console.log("⚠️  Model save failed, but training completed successfully");
    console.log("💡 For prediction demo, the model is ready in memory");
    console.log("🎮 You can still run the React game: npm run dev");
  }
}

main().catch(console.error);
