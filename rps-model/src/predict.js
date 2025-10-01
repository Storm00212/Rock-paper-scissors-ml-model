import * as tf from '@tensorflow/tfjs';
const CLASSES = ['rock', 'paper', 'scissors'];
let model;
async function loadModel() {
    model = await tf.loadLayersModel('./rps-model/model.json');
    console.log('Model loaded');
}
async function predict(imageElement) {
    if (!model)
        await loadModel();
    const tensor = tf.browser.fromPixels(imageElement)
        .resizeNearestNeighbor([64, 64])
        .toFloat()
        .div(255.0)
        .expandDims(0);
    const prediction = model.predict(tensor);
    const probabilities = prediction.dataSync();
    const predictedClass = CLASSES[probabilities.indexOf(Math.max(...probabilities))];
    return { predictedClass, probabilities: Array.from(probabilities) };
}
async function setupCamera() {
    const video = document.createElement('video');
    video.width = 320;
    video.height = 240;
    video.autoplay = true;
    document.body.appendChild(video);
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    document.body.appendChild(canvas);
    const captureButton = document.createElement('button');
    captureButton.textContent = 'Capture and Predict';
    document.body.appendChild(captureButton);
    const resultDiv = document.createElement('div');
    document.body.appendChild(resultDiv);
    captureButton.addEventListener('click', async () => {
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, 64, 64);
        const img = new Image();
        img.src = canvas.toDataURL();
        img.onload = async () => {
            const result = await predict(img);
            resultDiv.textContent = `Prediction: ${result.predictedClass}`;
        };
    });
}
async function main() {
    await loadModel();
    await setupCamera();
}
main().catch(console.error);
//# sourceMappingURL=predict.js.map