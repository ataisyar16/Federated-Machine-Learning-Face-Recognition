const express = require("express");
const cors = require("cors");
const multer = require("multer");
const canvas = require("canvas");
const faceapi = require("face-api.js");
const { Canvas, Image, ImageData } = canvas;

faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const app = express();

// Middleware
app.use(cors());
app.use(express.json());

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "./uploads");
  },
  filename: function (req, file, cb) {
    cb(null, `${Date.now()}_${file.originalname}`);
  },
});

const upload = multer({ storage });

// Load face-api models
async function loadModels() {
  const modelPath = "./models";
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);
}

loadModels().then(() => {
  console.log("Models loaded successfully");
});

app.post("/upload", upload.single("file"), async (req, res) => {
  if (!req.file) {
    return res.status(400).send("No file uploaded.");
  }

  try {
    const img = await canvas.loadImage(req.file.path);
    const detections = await faceapi
      .detectAllFaces(img)
      .withFaceLandmarks()
      .withFaceDescriptors();

    console.log(detections);
    res.json(detections);
  } catch (error) {
    console.error(error);
    res.status(500).send("Error processing the image.");
  }
});

console.log("Server starting...");
const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
