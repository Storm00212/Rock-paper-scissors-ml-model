# ✊✋✌️ Rock–Paper–Scissors AI (TypeScript + TensorFlow.js)

This project trains a **machine learning model** to recognize **Rock, Paper, and Scissors** hand gestures using  **images** .

It is built with **TypeScript** and  **TensorFlow.js** , and can run in **Node.js** for training and in the **browser** for real-time webcam prediction.

---

## 📂 Project Structure

```
rps-ai/
  dataset/              # Training images (rock/, paper/, scissors/)
  rps-model/            # Trained model artifacts
  src/
    train.ts            # Model training script (Node.js)
    predict.ts          # Prediction logic (Node.js/Browser)
  package.json
  tsconfig.json
  .gitignore
  README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone & Install

```bash
git clone https://github.com/your-username/rps-ai.git
cd rps-ai
npm install
```

### 2. Prepare Dataset

Organize images into folders:

```
dataset/
  rock/
    rock1.jpg
    rock2.jpg
    ...
  paper/
    paper1.jpg
    ...
  scissors/
    scissor1.jpg
    ...
```

### 3. Train Model

```bash
npm run train
```

👉 This saves the trained model into `rps-model/`.

### 4. Run Prediction

```bash
npm run predict
```

👉 Loads the model and predicts from test images (or webcam in browser setup).

---

## 📷 Webcam Prediction (Browser)

To use your webcam:

1. Bundle the project with **Vite** or  **Next.js** .
2. Import the saved model (`rps-model/`).
3. Use `navigator.mediaDevices.getUserMedia()` to access the camera.
4. Convert webcam frames into tensors for prediction.

---

## 🛠 Requirements

* Node.js 18+
* TypeScript 5+
* TensorFlow.js 4.22+

---

## 🚀 Future Improvements

* Add real-time webcam prediction UI
* Improve accuracy with data augmentation
* Deploy as a web app for live gameplay

---

## 📜 License

MIT © 2025 Paul Muyali
