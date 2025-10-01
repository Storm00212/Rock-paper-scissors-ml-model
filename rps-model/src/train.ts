import * as tf from "@tensorflow/tfjs";
import * as fs from "fs";
import * as path from "path";
import sharp from "sharp";

const DATASET_PATH = path.join(__dirname, "../../dataset");
const CLASSES: string[] = ["rock", "paper", "scissors"];

async function loadImages(): Promise<{ xs: tf.Tensor4D; ys: tf.Tensor2D }> {
  const images: tf.Tensor4D[] = [];
  const labels: number[] = [];

  for (let i = 0; i < CLASSES.length; i++) {
    const className: string = CLASSES[i]!;
    const dir = path.join(DATASET_PATH, className);

    if (!fs.existsSync(dir)) throw new Error(`Dataset folder not found: ${dir}`);

    const files = fs.readdirSync(dir);

    for (const file of files) {
      if (!file.endsWith(".png")) continue;

      const imgBuffer = fs.readFileSync(path.join(dir, file));
      const resizedBuffer = await sharp(imgBuffer).resize(64, 64).raw().toBuffer();
      const imgTensor = tf.tensor3d(new Uint8Array(resizedBuffer), [64, 64, 3], 'float32').div(255.0) as tf.Tensor3D;

      images.push(imgTensor.expandDims(0) as tf.Tensor4D);
      labels.push(i);
    }
  }

  const xs = tf.concat(images) as tf.Tensor4D;
  const ys = tf.oneHot(tf.tensor1d(labels, "int32"), CLASSES.length) as tf.Tensor2D;

  return { xs, ys };
}

function createModel() {
  const model = tf.sequential();

  model.add(tf.layers.conv2d({ inputShape: [64, 64, 3], filters: 16, kernelSize: 3, activation: "relu" }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: "relu" }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: CLASSES.length, activation: "softmax" }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

async function main() {
  await tf.setBackend('cpu');
  await tf.ready();

  console.log("Loading dataset...");
  const { xs, ys } = await loadImages();
  console.log(`Dataset loaded: ${xs.shape[0]} samples`);

  const model = createModel();
  model.summary();

  console.log("Training model...");
  await model.fit(xs, ys, {
    epochs: 10,
    batchSize: 16,
    validationSplit: 0.2,
  });

  console.log("Saving model...");
  await model.save("file://./rps-model");
  console.log("✅ Training complete, model saved at ./rps-model");
}

main().catch(console.error);
