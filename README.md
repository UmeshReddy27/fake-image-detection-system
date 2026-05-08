# Fake Image Detection System

## 📌 Overview

The Fake Image Detection System is a dual-branch, enterprise-grade machine learning pipeline engineered to detect synthetic media, diffusion-generated art, and deepfake manipulations. Rather than relying on a single generalized model, this system utilizes **Dynamic Computer Vision Routing** to inspect the input, detect facial boundaries, and seamlessly route the data to specialized neural networks.

To ensure transparency, the pipeline is integrated with **Explainable AI (XAI)**, generating real-time thermal heatmaps to visually demonstrate the neural network's decision-making logic.

## 🚀 Key Architectural Features

* **Dual-Branch CNN Inference:** * **Branch A (Facial Deepfakes):** Utilizes an `EfficientNetB0` backbone, fine-tuned to detect micro-manipulations, asymmetrical artifacts, and synthetic blending in human faces.
* **Branch B (General AI Art):** Utilizes a `MobileNetV3` backbone trained on the CIFAKE dataset to detect diffusion noise and pixel-level generation patterns in landscapes, objects, and general media.


* **Dynamic MTCNN Routing:** Implements an automated pre-processing router using Multi-Task Cascaded Convolutional Networks (MTCNN). If a face is detected, the system calculates a 30% dynamic padding boundary, crops the ROI (Region of Interest), and routes it to Branch A. If no face is detected, the full image is routed to Branch B.
* **Explainable AI (Grad-CAM):** Bypasses the "black box" nature of deep learning by slicing the functional models at the final convolutional layer. It calculates Gradient-weighted Class Activation Mapping (Grad-CAM) to project a thermal heatmap over the original image, highlighting the exact pixel clusters that influenced the final verdict.
* **Stable Inference Architecture:** Engineered using the TensorFlow Functional API with explicit inference-mode hardwiring (`training=False`) and weights-only `.h5` injection, guaranteeing absolute mathematical stability and avoiding common Keras 3 serialization graph corruptions.
* **Premium Glassmorphism UI:** Features a responsive, dark-theme asynchronous dashboard built with Tailwind CSS. Includes drag-and-drop mechanics, animated confidence meters, and sharp pixel-rendering for low-resolution dataset analysis.

## 🧠 The Pipeline Flow

1. **Ingestion:** User uploads an image via the asynchronous Flask frontend.
2. **Routing (MTCNN):** The image is converted to RGB and scanned for facial keypoints.
3. **Pre-Processing:** The image is mathematically reshaped, padded, and cast to `float32` tensors.
4. **Inference:** The tensor is passed through the highly-specialized neural network branch.
5. **Gradient Tracking:** The XAI module calculates the gradients of the target class relative to the feature map activations.
6. **Synthesis:** The backend compiles the confidence percentages, applies the Jet color map to the gradients, and returns the superimposed forensic package to the UI.

## 🛠️ Technology Stack

* **Machine Learning:** TensorFlow, Keras (Functional API)
* **Computer Vision:** OpenCV (`cv2`), MTCNN
* **Explainable AI:** Grad-CAM, Matplotlib (Color Mapping)
* **Backend / Server:** Python, Flask, Werkzeug
* **Frontend:** HTML5, JavaScript (Fetch API), Tailwind CSS

## 💻 Local Setup & Installation

**1. Clone the repository**

```bash
git clone https://github.com/yourusername/fake-image-detection-system.git
cd fake-image-detection-system

```

**2. Create a Virtual Environment & Install Dependencies**

```bash
python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
pip install tensorflow opencv-python mtcnn flask matplotlib numpy

```

**3. Run the Training Scripts (Generates Weights)**
Ensure your datasets are in the root directory, then initialize the memory configurations:

```bash
python train.py          # Trains Branch A (Faces)
python train_general.py  # Trains Branch B (General Media)

```

**4. Boot the Inference Server**

```bash
python app.py

```

*The dashboard will now be live at `http://127.0.0.1:5000`.*

---

* Implemented dynamic computer vision routing via **MTCNN** for automated facial detection, intelligent bounding-box padding, and precise tensor preprocessing.
* Architected a highly stable inference engine using the **TensorFlow Functional API** and weights-only injection to bypass complex serialization graph corruptions.
* Integrated **Explainable AI (XAI) using Grad-CAM** to perform gradient-tracking and generate real-time thermal heatmaps, providing transparent visual evidence of the model’s decision-making logic on a responsive **Tailwind CSS** dashboard.
